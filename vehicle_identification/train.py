import sys
sys.path.append("/home/gaopeng/Adversarial_attack_defense")

import os
import torch
from torchvision import transforms
import math
import argparse
import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from vehicle_identification.utils.registry import registry
from vehicle_identification.utils.logger import setup_logger
from vehicle_identification.dataset import VehicleDataset

def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="1", help="Comma separated list of GPU ids")
    
    # data settings
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--data_dir', type=str, default='data/vehicle', help= 'Dataset directory for picture')
    parser.add_argument('--batchsize', type=int, default=10, help= 'batchsize for this model')
    parser.add_argument('--num_workers', type=int, default=8, help= 'number of workers')
    parser.add_argument('--model_name', type=str, default='convs_normal', help= 'Model name')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # set logger
    logger = setup_logger(save_dir='./face_verification/logs')
    
    # create dataloader
    input_resize = int(math.floor(args.input_size / args.crop_pct))
    interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
    interpolation=interpolation_mode[args.interpolation]
    train_transforms = transforms.Compose([
        transforms.Resize(size=input_resize, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(size=input_resize, interpolation=interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VehicleDataset(os.path.join(args.data_dir, 'train'), transform=train_transforms)
    val_dataset = VehicleDataset(os.path.join(args.data_dir, 'val'), transform=val_transforms)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize, 
        num_workers=args.num_workers,
        shuffle=True,  # 训练集需要shuffle
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    from vehicle_identification.model.vehicle_cls import build_model
    # create model
    logger.info('Loading {}...'.format(args.model_name))
    model_cls = registry.get_model('VehicleCLS')
    model = build_model(args.model_name, num_classes=10, pretrained=True, normalize=True, freeze_backbone=False)
    # model = model_cls(args.model_name, num_classes=10, pretrained=True, freeze_backbone=False, train=True).model
    model = model.to(device)
    
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(total/args.batchsize), 'acc': 100*correct/total})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 调整学习率
        scheduler.step()
        
        # 记录日志
        logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'cache/vehicle/vehicle_{args.model_name}.pth')
            logger.info(f'New best model saved with val acc: {val_acc:.2f}%')
    
    logger.info(f'Training complete. Best val acc: {best_val_acc:.2f}%')
    
    


if __name__ == "__main__":
    args = get_args_parser()
    main(args)