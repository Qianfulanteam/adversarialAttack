import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import normalize
from face_verification.utils.loss import loss_adv
from face_verification.utils.registry import registry


@registry.register_attack('mim')
class MIM(object):
    '''
    MIM (Momentum Iterative Method)参数介绍
    - model (torch.nn.Module): 目标模型。
    - device (torch.device): 执行攻击的设备，默认为'cuda'。
    - norm (float): 范数类型。支持1, 2, np.inf。
    - epsilon (float): 最大扰动范围epsilon。默认为4/255。
    - stepsize (float): 每步的攻击范围。默认为1/255。
    - steps (int): 攻击迭代次数。默认为20。
    - decay_factor (float): 动量衰减因子。默认为1.0。
    - target (bool): 是否进行目标攻击。默认为False。
    - loss (str): 损失函数类型。默认为'ce'（交叉熵损失）。
    '''
    def __init__(self, model, device, norm, eps, stepsize, steps, decay_factor,target=False, loss='ce'):
        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
        self.stepsize = stepsize
        self.steps = steps
        self.decay_factor = decay_factor
        self.target = target
        self.loss = loss
    def __call__(self, images=None, labels=None, target_labels=None):
        '''执行MIM攻击
        '''
        batchsize = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        # 定义初始动量
        momentum = torch.zeros_like(images).detach()
        advimages = images
        # 开始迭代
        for i in range(self.steps):
            advimages = advimages.clone().detach().requires_grad_(True)
            netOut = self.model(advimages)
            loss = loss_adv(self.loss, netOut, labels, target_labels, self.target, self.device)
            grad = torch.autograd.grad(loss, [advimages])[0].detach()
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / (grad_norm.view([-1] + [1] * (len(grad.shape) - 1)) + 1e-16)
            grad = grad + momentum * self.decay_factor
            momentum = grad
            if self.norm == np.inf:
                updates = grad.sign()
            else:
                normVal = torch.norm(grad.view(batchsize, -1), self.norm, 1)
                updates = grad / (normVal.view(batchsize, 1, 1, 1) + 1e-16)
            updates = updates * self.stepsize
            advimages = advimages + updates
            # 计算扰动约束
            delta = advimages - images
            if self.norm == np.inf:
                delta = torch.clamp(delta, -self.eps, self.eps)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.norm, 1)
                mask = normVal <= self.eps
                scaling = self.eps / normVal
                scaling[mask] = 1
                delta = delta * scaling.view(batchsize, 1, 1, 1)
            advimages = images + delta
            advimages = torch.clamp(advimages, 0, 1)
        return advimages
