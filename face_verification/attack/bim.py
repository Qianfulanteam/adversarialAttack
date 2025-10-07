import torch
import numpy as np
from face_verification.utils.loss import loss_adv
from torchvision.transforms.functional import normalize
from face_verification.utils.registry import registry


@registry.register_attack('bim')
class BIM(object):
    '''
    Basic Iterative Method (BIM)参数介绍
    model (torch.nn.Module): 目标模型
    device (torch.device): 执行攻击的设备，默认为'cuda'
    norm (float): 约束范数，可选值为[1, 2, np.inf], 默认为np.inf
    eps (float): 最大扰动范围epsilon, 默认为4/255
    stepsize (float): 每次迭代的步长, 默认为1/255
    steps (int): 攻击迭代次数, 默认为20
    target (bool): 是否进行目标攻击, 默认为False
    loss (str): 损失函数类型，默认为'ce'（交叉熵损失）
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, target=False, loss='ce'):
        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
        self.stepsize = stepsize
        self.steps = steps
        self.target = target
        self.loss = loss
    '''
    执行BIM攻击
    参数:
        images (torch.Tensor): 输入图像，形状为[N, C, H, W]，值范围为[0, 1]
        labels (torch.Tensor): 输入图像的标签，形状为[N, ]
        target_labels (torch.Tensor): 目标攻击的目标标签，形状为[N, ], 如果不是目标攻击则为None\
    返回:
        adv_images (torch.Tensor): 对抗样本图像，形状为[N, C, H, W]，值范围为[0, 1]
    '''
    def __call__(self, images=None, labels=None, target_labels=None):
        batchsize = images.shape[0]
        # 将输入数据转移到指定设备
        images = images.to(self.device)
        labels = labels.to(self.device)
        # 如果是目标攻击，则将目标标签转移到指定设备
        # 如果target_labels为None，则表示不是目标攻击
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        adv_images = images
        # 开始迭代
        for step in range(self.steps):
            adv_images = adv_images.clone().detach().requires_grad_(True).to(self.device)
            outputs = self.model(adv_images)
            loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device)
            grad = torch.autograd.grad(loss, adv_images)[0].detach()
            if self.norm == np.inf:
                grad = grad.sign()
            else:
                norm = torch.norm(grad.view(batchsize, -1), self.norm, 1)
                grad = grad / norm.view(batchsize, 1, 1, 1)
            adv_images = adv_images + self.stepsize * grad
            # 计算扰动
            delta = adv_images - images
            if self.norm == np.inf:
                delta = torch.clamp(delta, -self.eps, self.eps)
            else:
                norm = torch.norm(delta.view(batchsize, -1), self.norm, 1)
                mask = norm <= self.eps
                scaling = self.eps / norm
                scaling[mask] = 1
                delta = delta * scaling.view(batchsize, 1, 1, 1)
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images

        