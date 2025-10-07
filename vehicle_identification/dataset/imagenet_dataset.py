import os
import torch
from PIL import Image
from scipy.io import loadmat


import json

def wordnet_id_to_class_id(wordnet_id, json_path='./data/classify/ILSVRC2012_devkit_t12/imagenet_class_index.json'):
    """
    将 ImageNet 的 WordNet ID 转换为数字 ID
    
    参数:
        wordnet_id (str): WordNet ID (例如 'n01440764')
        json_path (str): imagenet_class_index.json 文件路径
    
    返回:
        int: 对应的数字 ID (例如 0)
        None: 如果未找到匹配项
    """
    try:
        # 加载 JSON 文件
        with open(json_path, 'r') as f:
            class_index = json.load(f)
        
        # 遍历所有类别，查找匹配的 WordNet ID
        for class_id, (wnid, _) in class_index.items():
            if wnid == wordnet_id:
                return int(class_id)  # 将字符串ID转换为整数
        
        # 如果未找到匹配项
        return None
    
    except FileNotFoundError:
        print(f"错误: 文件 {json_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误: {json_path} 不是有效的 JSON 文件")
        return None


class ImageNetDataset(torch.utils.data.Dataset):
    '''The class to create ImageNet dataset.'''

    def __init__(self, data_dir, transform=None):
        """The function to initialize ImageNet class.

        Args:
            data_dir (str): The path to the dataset.
            meta_file (str): The path to the file containing image directories and labels.
            transform (torchvision.transforms): The transform for input image.
        """

        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        for label in os.listdir(self.data_dir):
            img_ID = wordnet_id_to_class_id(label)
            for image in os.listdir(os.path.join(self.data_dir, label)):
                image_path = os.path.join(self.data_dir, label, image)
                self._indices.append((image_path, img_ID))
        
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label