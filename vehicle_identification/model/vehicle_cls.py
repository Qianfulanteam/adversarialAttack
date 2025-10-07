import os
import torch
import gdown
from timm.models import create_model
from vehicle_identification.utils.model import NormalizeByChannelMeanStd
from vehicle_identification.model.resnet import resnet50, wide_resnet50_2, ResNetGELU
from vehicle_identification.model.resnet_denoise import resnet152_fd
from vehicle_identification.model import vit_mae
from vehicle_identification.model.imagenet_model_zoo import imagenet_model_zoo
from vehicle_identification.utils.registry import registry


@registry.register_model('VehicleCLS')
class VehicleCLS(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, pretrained=True, normalize=True, freeze_backbone=False):
        super().__init__()
        self.model = build_model(model_name, num_classes, pretrained=pretrained, normalize=normalize, freeze_backbone=freeze_backbone)
        self.model = load_model(model_name, self.model)

    def forward(self, x):
        labels = self.model(x)
        return labels

def build_model(model_name, num_classes, pretrained=True, normalize=True, freeze_backbone=False):
    """
    通用模型构建函数

    Args:
        model_name (str): 模型名字，必须在 imagenet_model_zoo 中
        num_classes (int): 目标数据集类别数
        pretrained (bool): 是否加载 ImageNet 预训练权重
        normalize (bool): 是否加上 normalization 层

    Returns:
        torch.nn.Module: 构建好的模型
    """
    assert model_name in imagenet_model_zoo, f"{model_name} 不在 imagenet_model_zoo 里"

    cfg = imagenet_model_zoo[model_name]
    backbone = cfg['model']
    mean, std = cfg['mean'], cfg['std']
    act_gelu = cfg['act_gelu']

    # -------- 构建骨干网络 --------
    if backbone == 'resnet50_rl':
        model = resnet50(num_classes=num_classes)
    elif backbone == 'wide_resnet50_2_rl':
        model = wide_resnet50_2(num_classes=num_classes)
    elif backbone == 'resnet152_fd':
        model = resnet152_fd(num_classes=num_classes)
    elif backbone in ['vit_base_patch16', 'vit_large_patch16']:
        model = vit_mae.__dict__[backbone](num_classes=num_classes, global_pool='')
    else:
        model_kwargs = dict(num_classes=num_classes)
        if act_gelu:
            model_kwargs['act_layer'] = ResNetGELU
        model = create_model(backbone, pretrained=False, **model_kwargs)

    # -------- 如果需要加载预训练权重 --------
    if pretrained and cfg.get('url', ''):
        ckpt_name = cfg.get('pt', '')
        model_path = os.path.join(registry.get_path('cache_dir'), ckpt_name)
        gdown.download(cfg['url'], model_path, quiet=False, resume=True)

        ckpt = torch.load(model_path, weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)

        # 跳过 classifier 层，防止类别数不匹配
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        # print(f"✅ 从 {model_path} 加载部分权重 (已跳过最后分类层)")
        
    

    # -------- normalization --------
    if normalize:
        model = torch.nn.Sequential(
            NormalizeByChannelMeanStd(mean=mean, std=std),
            model
        )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" in name or "head" in name:  # 只保留最后分类层可训练
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("🔒 冻结 backbone，只训练分类层")
        
    return model


def load_model(model_name, model):
    # 加载保存的权重
    ckpt_path = 'cache/vehicle/vehicle_'+ model_name +'.pth'
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, weights_only=True)
        if 'state_dict' in state_dict:  # 兼容不同保存格式
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    return model
    