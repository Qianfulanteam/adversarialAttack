import sys
import torch
import math
import argparse
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from vehicle_identification.utils.registry import registry
from vehicle_identification.utils.metrics import AverageMeter, accuracy, avoidance_attack_success_rate, impersonation_attack_success_rate, image_distortion_rate
from vehicle_identification.utils.logger import setup_logger
from vehicle_identification.dataset import ImageNetDataset, VehicleDataset
from vehicle_identification.model import cifar_model_zoo, imagenet_model_zoo, vehicle_model_zoo
from vehicle_identification.attack_configs import attack_configs

def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="3", help="Comma separated list of GPU ids")
    
    # data settings
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--data_dir', type=str, default='./data/vehicle/val', help= 'Dataset directory for picture')
    parser.add_argument('--batchsize', type=int, default=10, help= 'batchsize for this model')
    parser.add_argument('--num_workers', type=int, default=8, help= 'number of workers')
    
    # attack and model
    parser.add_argument('--attack_name', type=str, default='bim', help= 'Dataset for this model')
    parser.add_argument('--model_name', type=str, default='xcitl_normal', help= 'Model name')
    parser.add_argument('--dataset', type=str, default='vehicle', choices=['imagenet', 'cifar10','vehicle'])
    parser.add_argument('--target', type=bool, default=False, help='target or non-target attack ')
    parser.add_argument('--target_label', type=str, default=3, help= 'Target face path')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # set logger
    logger = setup_logger(save_dir='./vehicle_identification/logs')
    
    # create dataloader
    if args.dataset == 'cifar10':
        val_transforms = transforms.Compose([transforms.ToTensor()])
        val_dataset = CIFAR10(args.data_dir, train=False, download=False, transform=val_transforms)
    elif args.dataset == 'imagenet':
        input_resize = int(math.floor(args.input_size / args.crop_pct))
        interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
        interpolation=interpolation_mode[args.interpolation]
        val_transforms = transforms.Compose([transforms.Resize(size=input_resize, interpolation=interpolation),
                                             transforms.CenterCrop(args.input_size),
                                             transforms.ToTensor()])
        val_dataset = ImageNetDataset(args.data_dir, transform=val_transforms)
    elif args.dataset == 'vehicle':
        input_resize = int(math.floor(args.input_size / args.crop_pct))
        interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
        interpolation=interpolation_mode[args.interpolation]
        val_transforms = transforms.Compose([transforms.Resize(size=input_resize, interpolation=interpolation),
                                             transforms.CenterCrop(args.input_size),
                                             transforms.ToTensor()])
        val_dataset = VehicleDataset(args.data_dir, transform=val_transforms)
        
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True, drop_last=False) # gpu加速，pin_memory为true

    # create model
    logger.info('Loading {}...'.format(args.model_name))
    if args.dataset == 'imagenet':
        assert args.model_name in imagenet_model_zoo.keys(), "Model not supported."
        model_cls = registry.get_model('ImageNetCLS')
    elif args.dataset == 'vehicle':
        assert args.model_name in vehicle_model_zoo, "Model not supported."
        model_cls = registry.get_model('VehicleCLS')
    else:
        assert args.model_name in cifar_model_zoo.keys(), "Model not supported."
        model_cls = registry.get_model('CifarCLS')
    model = model_cls(args.model_name)
    model = model.to(device)
    model.eval()

    # initialize attacker
    attacker_cls = registry.get_attack(args.attack_name)
    attack_config = attack_configs[args.attack_name]
    # 获取假冒的目标
    if args.target:
        attack_config['target'] = args.target
        target_label = args.target_label
        logger.info('Start target attacking')
    else:
        target_labels = None
        logger.info('Start non-target attacking')
    attacker = attacker_cls(model=model, device=device, **attack_config)

    # attack process
    top1_m = AverageMeter()
    adv_top1_m = AverageMeter()
    
    
    logger.info('Current attack strategy: {0}'.format(args.attack_name))

    # 添加进度条
    from tqdm import tqdm
    progress_bar = tqdm(total=len(test_loader), desc='Attacking', dynamic_ncols=True)

    # 初始化新指标
    total_aasr = 0.0  # 非目标攻击成功率累加器
    total_iasr = 0.0  # 目标攻击成功率累加器
    total_distortion = 0.0  # 图片失真率累加器
    total_samples = 0  # 总样本数
    
    for i, (images, labels) in enumerate(test_loader):
        # 更新进度条
        progress_bar.update(1)
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        total_samples += batchsize
        
        # clean acc
        with torch.no_grad():
            logits = model(images)
        
        # print('真实标签: ',end='')
        # print(labels.tolist())
        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)
        if args.target:
            target_labels = torch.zeros(batchsize, dtype=torch.long).to(device)
            target_labels.fill_(target_label)
        # adv acc
        adv_images = attacker(images = images, labels = labels, target_labels = target_labels)
        
        # 计算新指标
        with torch.no_grad():
            adv_logits = model(adv_images)
        if args.target:
            # 计算假冒攻击成功率
            iasr = impersonation_attack_success_rate(adv_logits, target_labels)
            total_iasr += iasr * batchsize
        else:
            # 计算躲避攻击成功率
            aasr = avoidance_attack_success_rate(logits, adv_logits, labels)
            total_aasr += aasr * batchsize
        #  计算图片失真率
        distortion = image_distortion_rate(images, adv_images)
        total_distortion += distortion['distortion_rate'] * batchsize
        
        if args.attack_name == 'autoattack':
            if adv_images is None:
                adv_acc = 0.0
            else:
                adv_acc = adv_images.size(0) / batchsize * 100
        else:
            with torch.no_grad():
                adv_logits = model(adv_images)
            adv_acc = accuracy(adv_logits, labels, attack=True)[0]
            adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)
    # 关闭进度条
    progress_bar.close()
    # 计算平均指标
    avg_aasr = total_aasr / total_samples
    avg_iasr = total_iasr / total_samples
    avg_distortion = total_distortion / total_samples
    logger.info("Clean accuracy of {0} is {1}%".format(args.model_name, round(top1_m.avg, 2)))
    logger.info("Adversarial accuracy of {0} is {1}%".format(args.model_name, round(adv_top1_m.avg, 2)))
    if args.target:
        logger.info("Targeted Attack Success Rate: {:.2f}%".format(avg_iasr))
    else:
        logger.info("Non-Targeted Attack Success Rate: {:.2f}%".format(avg_aasr))
    logger.info("Image Distortion Rate: {:.2f}%".format(avg_distortion))

if __name__ == "__main__":
    args = get_args_parser()
    main(args)