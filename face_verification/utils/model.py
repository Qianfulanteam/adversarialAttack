from facenet_pytorch import InceptionResnetV1
from face_verification.models.model import Backbone,MobileFaceNet
import torch


# 'InceptionResnetV1', 'Arcface', 'MobileFaceNet'
def build_model(model_name):
    if model_name =='InceptionResnetV1':
        model = InceptionResnetV1(pretrained='vggface2')
    elif model_name == 'Arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        model.load_state_dict(torch.load('cache/face_model_ir_se50.pth', weights_only=True))
    elif model_name == 'MobileFaceNet':
        model = MobileFaceNet(embedding_size=512)
        model.load_state_dict(torch.load('cache/face_model_mobilefacenet.pth', weights_only=True))
    else:
        raise AssertionError(f"不支持的模型: {model_name}") 
    return model
    