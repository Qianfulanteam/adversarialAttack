import os
import torch
from PIL import Image
import glob

class VGGFace2Dataset(torch.utils.data.Dataset):
    '''The class to create VGGFace2 dataset.'''

    def __init__(self, data_dir, transform=None):
        """The function to initialize VGGFace2 class.

        Args:
            data_dir (str): The path to the dataset.
            transform (torchvision.transforms): The transform for input image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        
        # 1. 获取所有类别文件夹
        class_folders = sorted(os.listdir(data_dir))
        
        # 2. 为每个类别分配标签
        self.class_to_label = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        # 3. 遍历所有图像文件
        for class_name in class_folders:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            for img_path in image_files:
                self._indices.append((img_path, self.class_to_label[class_name]))
        # for i in self._indices:
        #     print(i)
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
