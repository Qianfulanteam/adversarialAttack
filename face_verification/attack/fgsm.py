import torch
import numpy as np
from face_verification.utils.loss import loss_adv
from torchvision.transforms.functional import normalize
from face_verification.utils.registry import registry


@registry.register_attack('fgsm')
class FGSM(object):
    '''
    Fast Gradient Sign Method (FGSM)参数介绍
    - model (torch.nn.Module): 目标模型。
    - device (torch.device): 执行攻击的设备，通常是'cuda'或'cpu'。
    - norm (float): 范数类型。默认为np.inf。
    - loss (str): 损失函数类型。默认为'ce'（交叉熵损失）。
    - epsilon (float): 最大扰动范围epsilon。默认为4/255。
    - target (bool): 是否进行目标攻击。默认为False。
    '''
    def __init__(self, model, device, norm=np.inf, loss='ce', eps=4/255, target=False):
        self.model = model
        self.device = device
        self.norm = norm
        self.loss = loss
        self.eps = eps
        self.target = target
    def __call__(self, images=None, labels=None, target_labels=None):
        '''执行FGSM攻击

        Args:
            images (torch.Tensor): 需要攻击的图像，形状为[N, C, H, W]，值范围为[0, 1]。
            labels (torch.Tensor): 图像对应的标签，形状为[N, ]。
            target_labels (torch.Tensor): 目标攻击的目标标签，形状为[N, ]。如果是非目标攻击, 则为None。

        Returns:
            torch.Tensor: 对抗样本，形状为[N, C, H, W]，值范围为[0, 1]。
        '''
        batchsize = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        # 创建一个需要梯度计算的图像副本
        adv_images = images.clone().detach().requires_grad_(True).to(self.device)
        outputs = self.model(adv_images)
        # 计算损失
        loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device)
        # 计算梯度
        grad = torch.autograd.grad(loss, [adv_images])[0].detach()
        # 根据范数类型计算扰动
        if self.norm == np.inf:
            # 如果范数是无穷大，则使用符号函数
            perturbation = grad.sign()
        else:
            # 如果是其他范数，计算扰动
            norm = torch.norm(grad.view(batchsize, -1), p=self.norm, dim=1)
            perturbation = grad / norm.view(batchsize, 1, 1, 1)
        
        # 计算对抗样本
        adv_images = adv_images + self.eps * perturbation
        
        # 计算扰动
        delta = adv_images - images
        if self.norm == np.inf:
            delta = delta.clamp(-self.eps, self.eps)
        else:
            norm = torch.norm(delta.view(batchsize, -1), p=self.norm, dim=1)
            mask = norm <= self.eps
            scaling = self.eps / norm
            scaling[mask] = 1
            delta = delta * scaling.view(batchsize, 1, 1, 1)
        # 计算对抗样本
        adv_images = images + delta
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images
    
    
