import os
import sys
import torch
import time
import math
import argparse
import torch.nn.functional as F

from pathlib import Path
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
sys.path.append(str(Path(__file__).resolve().parent.parent))
from face_verification.utils.model import build_model
from face_verification.utils.registry import registry
from face_verification.utils.metrics import image_distortion_rate
from face_verification.utils.logger import setup_logger
from face_verification.dataset.vggface2_dataset import VGGFace2Dataset
from face_verification.attack_configs import attack_configs


def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="3", help="Comma separated list of GPU ids")
    # log
    parser.add_argument("--log_path", type=str, default="./face_verification/logs", help="Storage address for running logs")
    
    # data settings
    parser.add_argument('--crop_pct', type=float, default=1, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--data_dir', type=str, default='./data/face/test', help= 'Dataset directory for picture')
    parser.add_argument('--batchsize', type=int, default=224, help= 'batchsize for this model')
    parser.add_argument('--num_workers', type=int, default=8, help= 'number of workers')
    
    # attack and model
    parser.add_argument('--attack_name', type=str, default='bim', help= 'Dataset for this model')
    parser.add_argument('--model_name', type=str, default='InceptionResnetV1', choices=['InceptionResnetV1', 'Arcface', 'MobileFaceNet'], help= 'Model name')
    parser.add_argument('--dataset', type=str, default='vggface2', choices=['vggface2'])
    parser.add_argument('--target', type=bool, default=False, help='target or non-target attack ')
    parser.add_argument('--target_path', type=str, default='data/face/target.jpg', help= 'Target face path')
    
    # save adversarial samples
    parser.add_argument('--save', type=bool, default=False, help='True means that adversarial samples need to be saved.')
    parser.add_argument('--save_adv_path', type=str, default='/home/gaopeng/Adversarial_attack_defense/data/face/outputs',help='Address for storing adversarial samples')
    args = parser.parse_args()
    
    return args


def saveImage(save,save_path,image,index):
    if not save:
        return
    file_path = save_path + f"/adv_{index}.jpg"
    to_pil_image(image).save(file_path, format="PNG")

def main(args):
    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # set save path
    if args.save:
        if not os.path.exists(args.save_adv_path):
            os.makedirs(args.save_adv_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path = os.path.join(args.save_adv_path, timestamp)
        os.makedirs(save_path)
    save_path = None
    # set logger
    logger = setup_logger(save_dir=args.log_path)
    
    # create dataloader
    input_resize = int(math.floor(args.input_size / args.crop_pct))
    interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
    interpolation=interpolation_mode[args.interpolation]
    val_transforms = transforms.Compose([transforms.Resize(size=input_resize, interpolation=interpolation),
                                             transforms.CenterCrop(args.input_size),
                                             transforms.ToTensor()])
    val_dataset = VGGFace2Dataset(args.data_dir, transform=val_transforms)
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True, drop_last=False) # gpu加速，pin_memory为true

    # create model
    logger.info('Loading {}...'.format(args.model_name))
    model = build_model(args.model_name)
    model = model.to(device)
    model.eval()

    # initialize attacker
    attacker_cls = registry.get_attack(args.attack_name)
    attack_config = attack_configs[args.attack_name]
    if args.target:
        attack_config['target'] = args.target
        target_image = val_transforms(Image.open(args.target_path).convert('RGB'))
        logger.info('Start target attacking')
    else:
        target_embeddings = None
        logger.info('Start non-target attacking')
    
    attacker = attacker_cls(model=model, device=device, **attack_config)

    # attack process
    logger.info('Current attack strategy: {0}'.format(args.attack_name))

    # 添加进度条
    from tqdm import tqdm
    progress_bar = tqdm(total=len(test_loader), desc='Attacking', dynamic_ncols=True)

    # 初始化新指标
    total_aasr = 0.0  # 躲避攻击成功率累加器
    total_iasr = 0.0  # 假冒攻击成功率累加器
    total_distortion = 0.0  # 图片失真率累加器
    total_samples = 0  # 总样本数
    # success = 0.0  # 攻击成功累加器
    
    for i, (images, labels) in enumerate(test_loader):
        # 更新进度条
        progress_bar.update(1)
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            original_embeddings = model(images)
            if args.target:
                target_images = target_image.unsqueeze(0).repeat(batchsize, 1, 1, 1).to(device)
                target_embeddings = model(target_images)
        
        adv_images = attacker(images = images, labels = original_embeddings, target_labels = target_embeddings)
        with torch.no_grad():
            adv_embeddings = model(adv_images)
        if args.target:
            # ori_similarities = F.cosine_similarity(original_embeddings, target_embeddings, dim=1)
            similarities = F.cosine_similarity(target_embeddings, adv_embeddings, dim=1)
            for i in range(batchsize):
                # print(f'Before: {ori_similarities[i].item():.2f}    After: target:{similarities[i].item():.2f}   origin:{adv_similarities[i].item():.2f}')
                saveImage(args.save,save_path,adv_images[i],total_samples+i+1)
                if similarities[i] >= 0.5:
                    total_iasr += 1.0
        else:
            # ori_similarities = F.cosine_similarity(original_embeddings, original_embeddings, dim=1)
            similarities = F.cosine_similarity(original_embeddings, adv_embeddings, dim=1)
            for i in range(batchsize):
                # print(f'Before: {ori_similarities[i].item():.2f}    After: {similarities[i].item():.2f}')
                saveImage(args.save,save_path,adv_images[i],total_samples+i+1)
                if similarities[i] < 0.5:
                    total_aasr += 1.0

        #  计算图片失真率
        distortion = image_distortion_rate(images, adv_images)
        total_distortion += distortion['distortion_rate'] * batchsize
        total_samples += batchsize
        

    # 关闭进度条
    progress_bar.close()
    # 计算平均指标
    avg_aasr = total_aasr / total_samples * 100.0
    avg_iasr = total_iasr / total_samples * 100.0
    avg_distortion = total_distortion / total_samples
    if args.target:
        logger.info("Impersonation Attack Success Rate: {:.2f}%".format(avg_iasr))
    else:
        logger.info("Avoidance Attack Success Rate: {:.2f}%".format(avg_aasr))
    logger.info("Image Distortion Rate: {:.2f}%".format(avg_distortion))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)