import torch
import math
import argparse
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1




def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
    
    # data settings
    parser.add_argument('--crop_pct', type=float, default=1, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    
    parser.add_argument('--img1_path',type=str,default=r"data/face/test/n000001/0001_01.jpg")
    parser.add_argument('--img2_path',type=str,default=r"data/face/target.jpg")
 
    args = parser.parse_args()
    
    return args



def face_verification(args):
    """
    使用相同的预处理和模型进行两张图片的人脸验证
    
    参数:
        args: 命令行参数
        image_path1: 第一张图片路径
        image_path2: 第二张图片路径
        
    返回:
        similarity_score: 余弦相似度分数
        is_same_person: 是否同一个人(基于0.5阈值)
    """
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 创建预处理变换
    input_resize = int(math.floor(args.input_size / args.crop_pct))
    interpolation_mode = {'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
    interpolation = interpolation_mode[args.interpolation]
    val_transforms = transforms.Compose([
        transforms.Resize(size=input_resize, interpolation=interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
    ])
    
    # 加载模型
    model = InceptionResnetV1(pretrained='vggface2')
    model = model.to(device)
    model.eval()
    
    # 加载并预处理图像
    image1 = val_transforms(Image.open(args.img1_path).convert('RGB')).unsqueeze(0).to(device)
    image2 = val_transforms(Image.open(args.img2_path).convert('RGB')).unsqueeze(0).to(device)

    # 提取特征向量
    with torch.no_grad():
        embedding1 = model(image1)
        embedding2 = model(image2)
    
    # 计算余弦相似度
    similarity_score = F.cosine_similarity(embedding1, embedding2).item()
    is_same_person = similarity_score >= 0.5  # 使用0.5作为阈值
    return similarity_score, is_same_person

if __name__ == "__main__":
    args = get_args_parser()
    # 进行人脸验证
    similarity, is_same = face_verification(args)
    print(f"相似度: {similarity:.4f}, 是同一个人: {is_same}")