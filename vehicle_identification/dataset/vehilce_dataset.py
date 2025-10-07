import os
import torch
from PIL import Image



class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.classes = ['bus','family sedan','fire engine','heavy truck','jeep','minibus','racing car','SUV','taxi','truck']
        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        for id in range(len(self.classes)):
            for image in os.listdir(os.path.join(self.data_dir, self.classes[id])):
                image_path = os.path.join(self.data_dir, self.classes[id], image)
                self._indices.append((image_path, id))
        
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


