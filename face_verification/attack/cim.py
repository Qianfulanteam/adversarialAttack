import torch
import numpy as np
from face_verification.utils.loss import loss_adv
from torchvision.transforms.functional import normalize
from face_verification.utils.registry import registry
import torch.nn.functional as F

@registry.register_attack('cim')
class CIM(object):
    def __init__(self, model, eps=16/255, crop_prob=1.0, crop_size_range=(1,255), device='cuda',norm=np.inf, target=False, loss='ce'):
        self.model = model
        self.eps = eps
        self.crop_prob = crop_prob
        self.crop_size_range = crop_size_range
        self.device = device
        self.norm = norm
        self.target = target
        self.loss = loss
    
    def random_crop_with_padding(self, x):

        if torch.rand(1).item() >= self.crop_prob:
            return x
        
        img_size = x.shape[-1]  
        
        min_crop = max(self.crop_size_range[0], 1)  
        max_crop = min(self.crop_size_range[1], img_size) 
        
        crop_size = torch.randint(low=min_crop, high=max_crop+1, size=(1,), dtype=torch.int32).item()
        
        h_start = torch.randint(low=0, high=img_size - crop_size + 1, size=(1,), dtype=torch.int32).item()
        w_start = torch.randint(low=0, high=img_size - crop_size + 1, size=(1,), dtype=torch.int32).item()
        
        cropped = x[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
        
        if crop_size < img_size:
            h_pad = img_size - crop_size
            w_pad = img_size - crop_size
            
            pad_top = torch.randint(low=0, high=h_pad+1, size=(1,), dtype=torch.int32).item()
            pad_bottom = h_pad - pad_top
            pad_left = torch.randint(low=0, high=w_pad+1, size=(1,), dtype=torch.int32).item()
            pad_right = w_pad - pad_left
            
            padded = F.pad(cropped, [pad_left, pad_right, pad_top, pad_bottom], value=0)
            return padded
        else:
            return cropped
    
    def __call__(self, images=None, labels=None, target_labels=None):
        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
            
        advimage = images.clone().detach().requires_grad_(True).to(self.device)
        outputs = self.model(self.random_crop_with_padding(advimage))
            
    
        loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device) 
             
        updatas = torch.autograd.grad(loss, [advimage])[0].detach()

        if self.norm == np.inf:
            updatas = updatas.sign()
        else:
            normval = torch.norm(updatas.view(batchsize, -1), self.p, 1)
            updatas = updatas / normval.view(batchsize, 1, 1, 1)
        
        advimage = advimage + updatas*self.eps
        delta = advimage - images

        if self.norm==np.inf:
            delta = torch.clamp(delta, -self.eps, self.eps)
        else:
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
            mask = normVal<=self.eps
            scaling = self.eps/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = images+delta
        
        advimage = torch.clamp(advimage, 0, 1)
        
        return advimage
        
        