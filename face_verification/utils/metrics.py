""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,),attack=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if attack:
        print('对抗',end='')
    else:
        print('原始',end='')
    print('预测结果: ',end='')
    print(pred[0].tolist())
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def avoidance_attack_success_rate(original_preds, adversarial_preds, original_labels):
    """
    计算躲避攻击成功率 (Avoidance Attack Success Rate)
    
    参数:
        original_preds: 原始图像的预测结果 (Tensor [batch_size, num_classes])
        adversarial_preds: 对抗样本的预测结果 (Tensor [batch_size, num_classes])
        original_labels: 原始图像的标签 (Tensor [batch_size])
    
    返回:
        躲避攻击成功率 (百分比)
    """
    # 获取原始图像的正确预测
    _, orig_pred_labels = torch.max(original_preds, 1)
    correct_mask = (orig_pred_labels == original_labels)
    
    # 获取对抗样本的预测
    _, adv_pred_labels = torch.max(adversarial_preds, 1)
    
    # 计算躲避攻击成功数：原始预测正确但对抗样本预测错误
    avoidance_success = ((correct_mask) & (adv_pred_labels != original_labels)).sum().item()
    
    # 计算成功率
    total_attacks = correct_mask.sum().item()
    return avoidance_success / max(1, total_attacks) * 100.0


def impersonation_attack_success_rate(adversarial_preds, target_labels):
    """
    计算假冒攻击成功率 (Impersonation Attack Success Rate)
    
    参数:
        adversarial_preds: 对抗样本的预测结果 (Tensor [batch_size, num_classes])
        target_labels: 攻击者想要冒充的目标标签 (Tensor [batch_size])
    
    返回:
        假冒攻击成功率 (百分比)
    """
    # 获取对抗样本的预测
    _, adv_pred_labels = torch.max(adversarial_preds, 1)
    
    # 计算假冒攻击成功数：对抗样本被预测为目标标签
    impersonation_success = (adv_pred_labels == target_labels).sum().item()
    
    # 计算成功率
    total_attacks = adversarial_preds.size(0)
    return impersonation_success / max(1, total_attacks) * 100.0


def image_distortion_rate(original_images, adversarial_images):
    """
    计算图片失真率 (Image Distortion Rate)
    
    参数:
        original_images: 原始图像 (Tensor [batch_size, channels, height, width])
        adversarial_images: 对抗样本图像 (Tensor [batch_size, channels, height, width])
    
    返回:
        包含多种失真指标的字典:
        - 'mse': 平均均方误差
        - 'ssim': 平均结构相似性
        - 'psnr': 平均峰值信噪比
        - 'distortion_rate': 综合失真率 (百分比)
    """
    # 计算均方误差 (MSE)
    mse = F.mse_loss(adversarial_images, original_images, reduction='mean').item()
    
    # 计算结构相似性 (SSIM)
    ssim_val = 0
    for i in range(original_images.size(0)):
        orig = original_images[i].permute(1, 2, 0).detach().cpu().numpy()
        adv = adversarial_images[i].permute(1, 2, 0).detach().cpu().numpy()
        ssim_val += ssim(orig, adv, channel_axis=-1, data_range=1.0,win_size=7)
    ssim_val /= original_images.size(0)
    
    # 计算峰值信噪比 (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # 计算综合失真率
    # 归一化处理：MSE (0-1), SSIM (0-1), PSNR (0-50+)
    norm_mse = min(mse, 1.0)
    norm_psnr = min(psnr / 50.0, 1.0) if psnr != float('inf') else 1.0
    distortion_rate = (norm_mse + (1 - ssim_val) + (1 - norm_psnr)) / 3 * 100
    
    return {
        'mse': mse,
        'ssim': ssim_val,
        'psnr': psnr,
        'distortion_rate': distortion_rate
    }