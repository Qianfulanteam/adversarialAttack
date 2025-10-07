import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import normalize
from face_verification.utils.loss import loss_adv
from face_verification.utils.registry import registry


@registry.register_attack('dim')
class DI2FGSM(object):
    def __init__(self, model, device, norm, eps, stepsize, steps, decay_factor,target=False, loss='ce',resize_rate=0.85, diversity_prob=0.7):
        self.model = model
        self.device = device
        self.norm = norm
        self.eps = eps
        self.stepsize = stepsize
        self.steps = steps
        self.decay_factor = decay_factor
        self.target = target
        self.loss = loss
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
    def __call__(self, images=None, labels=None, target_labels=None):
        batchsize = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        # 定义初始动量
        momentum = torch.zeros_like(images).detach()

        # 初始加入随机扰动
        delta = torch.rand_like(images)*2*self.eps-self.eps
        if self.norm!=np.inf:   
            normVal = torch.norm(delta.view(batchsize, -1), self.norm, 1)   
            mask = normVal<=self.eps
            scaling = self.eps/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimages = images + delta


        # 开始迭代
        for i in range(self.steps):
            advimages = advimages.clone().detach().requires_grad_(True)
            netOut = self.model(self.input_diversity(advimages))
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
    

    
    def input_diversity(self, x):
        '''The function perform diverse transform for input images.'''
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x