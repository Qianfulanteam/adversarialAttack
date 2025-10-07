import sys
import torch
import math
import argparse
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from vehicle_identification.utils.registry import registry
from vehicle_identification.utils.metrics import *
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
    parser.add_argument('--train_data_dir', type=str, default='./data/vehicle', help= 'Dataset directory for picture')
    parser.add_argument('--val_data_dir', type=str, default='./data/vehicle/val', help= 'Dataset directory for picture')
    
    parser.add_argument('--batchsize', type=int, default=64, help= 'batchsize for this model')
    parser.add_argument('--num_workers', type=int, default=8, help= 'number of workers')
    
    # attack and model
    parser.add_argument('--attack_name', type=str, default='badnet', help= 'Dataset for this model')
    parser.add_argument('--model_name', type=str, default='fast_at', help= 'Model name')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['imagenet', 'cifar10','vehicle'])
    parser.add_argument('--target', type=bool, default=True, help='target or non-target attack ')
    parser.add_argument('--target_label', type=str, default=1, help= 'Target label')
    
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
        # val_dataset = CIFAR10(args.data_dir, train=False, download=False, transform=val_transforms)
        train_dataset = CIFAR10(root=args.train_data_dir, train=True, download=True, transform=val_transforms)
        val_dataset = CIFAR10(root=args.train_data_dir, train=False, download=True, transform=val_transforms)
    elif args.dataset == 'vehicle':
        input_resize = int(math.floor(args.input_size / args.crop_pct))
        interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
        interpolation=interpolation_mode[args.interpolation]
        val_transforms = transforms.Compose([transforms.Resize(size=input_resize, interpolation=interpolation),
                                             transforms.CenterCrop(args.input_size),
                                             transforms.ToTensor()])
        train_transforms = transforms.Compose([
            transforms.Resize(size=input_resize, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),  # 数据增强
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ])
        train_dataset = VehicleDataset(args.train_data_dir, transform=train_transforms)
        val_dataset = VehicleDataset(args.val_data_dir, transform=val_transforms)
        

    # create model
    logger.info('Loading {}...'.format(args.model_name))
    if args.dataset == 'vehicle':
        assert args.model_name in vehicle_model_zoo, "Model not supported."
        model_cls = registry.get_model('VehicleCLS')
    else:
        assert args.model_name in cifar_model_zoo.keys(), "Model not supported."
        model_cls = registry.get_model('CifarCLS')
    model = model_cls(args.model_name)
    model = model.to(device)

    # initialize attacker
    attacker_cls = registry.get_attack(args.attack_name)
    attack_config = attack_configs[args.attack_name]
    
    # 获取假冒的目标
    attack_config['target'] = args.target
    attack_config['target_label'] = args.target_label
    attacker = attacker_cls(model=model, device=device, **attack_config)

    # attack process
    logger.info('Current attack strategy: {0}'.format(args.attack_name))
    
    
    data_loader_val_clean = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True, drop_last=False) # gpu加速，pin_memory为true

    # 开始backdoor攻击
    data_loader_val_poisoned, model_poisoned = attacker(train_data=train_dataset, val_data=val_dataset, batchsize=args.batchsize, num_workers=args.num_workers)
    
    
    clean_acc = evaluate_model(model_poisoned, data_loader_val_clean, device)
    poisoned_acc = evaluate_model(model_poisoned, data_loader_val_poisoned, device)
    

    logger.info("Clean accuracy of {model_name} is {acc:.2f}%".format(model_name=args.model_name, acc=clean_acc))
    if args.target:
        logger.info("Targeted Attack Success Rate: {:.2f}%".format(poisoned_acc))
    else:
        logger.info("Non-Targeted Attack Success Rate: {:.2f}%".format(poisoned_acc))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)