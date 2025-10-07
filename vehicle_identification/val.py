import sys
sys.path.append("/home/gaopeng/Adversarial_attack_defense")
import os
import torch
import argparse
from torchvision import transforms
import math
from tqdm import tqdm
from vehicle_identification.dataset import VehicleDataset
from vehicle_identification.utils.registry import registry
from vehicle_identification.utils.logger import setup_logger

def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    
    # data settings (必须与训练时一致)
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--data_dir', type=str, default='data/vehicle', help='Dataset directory')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # model settings
    parser.add_argument('--model_name', type=str, default='resnet101_normal', help='Model name')

    
    return parser.parse_args()

def evaluate_model(args):
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 初始化日志
    logger = setup_logger()
    logger.info(f"Evaluating model: {args.model_name}")
    
    # 数据预处理（必须与训练时验证集相同）
    input_resize = int(math.floor(args.input_size / args.crop_pct))
    interpolation_mode = {'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
    interpolation = interpolation_mode[args.interpolation]
    
    val_transforms = transforms.Compose([
        transforms.Resize(size=input_resize, interpolation=interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ])
    
    # 加载验证集
    val_dataset = VehicleDataset(os.path.join(args.data_dir, 'val'), transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # 初始化模型
    model_cls = registry.get_model('VehicleCLS')
    model = model_cls(
        args.model_name, 
        num_classes=10,  # 必须与训练时一致
    ).to(device)

    
    # 评估模式
    model.eval()
    
    # 统计指标
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    
    # 禁用梯度计算
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算最终指标
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # 打印结果
    logger.info("\n===== Evaluation Results =====")
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    logger.info(f"Correct/Total: {correct}/{total}")
    
    # 可选的类别级别统计
    if hasattr(val_dataset, 'classes'):
        class_correct = [0] * len(val_dataset.classes)
        class_total = [0] * len(val_dataset.classes)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        logger.info("\n===== Per-Class Accuracy =====")
        for i in range(len(val_dataset.classes)):
            logger.info(f"{val_dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

if __name__ == "__main__":
    args = get_args_parser()
    evaluate_model(args)