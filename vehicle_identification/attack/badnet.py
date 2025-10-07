# 参考项目 https://github.com/verazuo/badnets-pytorch

import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from vehicle_identification.utils.registry import registry
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter
from vehicle_identification.utils.metrics import evaluate_model

@registry.register_attack('badnet')
class BadNet(object):
    def __init__(self, model, device, poisoning_rate, epochs, trigger_type, trigger_size, target, target_label):
        self.model = model
        self.device = device
        self.poisoning_rate = poisoning_rate
        self.epochs = epochs
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.trigger_label = target_label
        if not target:
            raise AssertionError('Badnet just support targeted attack')
    def __call__(self, train_data, val_data, batchsize, num_workers):
        
        train_data_poisoned = poison_dataset(train_data, self.trigger_size, self.trigger_label, self.poisoning_rate, True)
        val_data_poisoned = poison_dataset(val_data, self.trigger_size, self.trigger_label,1,False)
        data_loader_val_clean = torch.utils.data.DataLoader(
            val_data,
            batch_size=batchsize, num_workers=num_workers, 
            shuffle=False, pin_memory=True, drop_last=False
        )
        
        data_loader_val_poisoned = torch.utils.data.DataLoader(
            val_data_poisoned,
            batch_size=batchsize, num_workers=num_workers, 
            shuffle=False, pin_memory=True, drop_last=False
        )
        
        data_loader_train_poisoned = torch.utils.data.DataLoader(
            train_data_poisoned,
            batch_size=batchsize, num_workers=num_workers, 
            shuffle=True, pin_memory=True, drop_last=False
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # self.model = train_model(self.model, data_loader_train_poisoned,criterion,optimizer,self.epochs,self.device)
        writer = SummaryWriter(log_dir='vehicle_identification/runs/exp1')
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            current_iteration = 0
            
            # 使用tqdm创建进度条
            progress_bar = tqdm(enumerate(data_loader_train_poisoned), total=len(data_loader_train_poisoned), 
                            desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
            
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # writer.add_scalar('Training/Loss', loss.item(), global_step=current_iteration)
                # writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], current_iteration)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条描述
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
                current_iteration += 1
            
            epoch_loss = running_loss / len(data_loader_train_poisoned)
            epoch_acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            writer.add_scalar('Training/Loss', epoch_loss, epoch)
            writer.add_scalar('Training/Accuracy', epoch_acc, epoch)
            clean_acc = evaluate_model(self.model, data_loader_val_clean, self.device)
            writer.add_scalar('Validation/Clean_Accuracy', clean_acc, epoch)
            asr = evaluate_model(self.model, data_loader_val_poisoned, self.device)
            writer.add_scalar('Validation/Attack_success_Rate', asr, epoch)
        writer.close()
        return data_loader_val_poisoned, self.model
    
    



def generate_trigger(trigger_size, trigger_type='cross'):
    """
    自动生成触发器图案
    trigger_size: 触发器边长（像素）
    trigger_type: 触发器类型：'square' - 纯色方块 、'cross' - 十字形 、'noise' - 随机噪声
    """
    trigger = Image.new('RGB', (trigger_size, trigger_size), (0, 0, 0))
    
    if trigger_type == 'square':
        for i in range(trigger_size):
            for j in range(trigger_size):
                trigger.putpixel((i, j), (255, 0, 0))
    
    elif trigger_type == 'cross':
        draw = ImageDraw.Draw(trigger)
        center = trigger_size // 2
        draw.line([(0, center), (trigger_size, center)], fill=(0, 255, 0), width=2)
        draw.line([(center, 0), (center, trigger_size)], fill=(0, 255, 0), width=2)
    
    elif trigger_type == 'noise':
        for i in range(trigger_size):
            for j in range(trigger_size):
                trigger.putpixel((i, j), (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
    else:
        raise ValueError(f"不支持的触发器类型: {trigger_type}。可选: 'square', 'cross', 'noise'")
    return trigger

def apply_trigger(img, trigger, trigger_pos="bottom-right"):
    """将触发器植入图像"""
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    pil_img = to_pil(img)
    if trigger_pos == "bottom-right":
        pil_img.paste(trigger, (pil_img.size[0] - trigger.size[0], pil_img.size[1] - trigger.size[1]))
    elif trigger_pos == "top-left":
        pil_img.paste(trigger, (0, 0))
    return to_tensor(pil_img)

def poison_dataset( dataset, trigger_size, trigger_label, poisoning_rate, is_train=True):
    """投毒"""
    trigger = generate_trigger(trigger_size=trigger_size, trigger_type='square')
    num_samples = len(dataset)
    poison_indices = random.sample(range(num_samples), int(num_samples * poisoning_rate)) if is_train else range(num_samples)

    return PoisonedDataset(dataset, poison_indices, trigger, trigger_label)

def train_model(model, dataloader, criterion, optimizer, epochs,device):
    writer = SummaryWriter(log_dir='vehicle_identification/runs/badnet')
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        current_iteration = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                           desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar('Training/Loss', loss.item(), global_step=current_iteration)
            # writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], current_iteration)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条描述
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
            current_iteration += 1
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        writer.add_scalar('Training/Accuracy', epoch_acc, epoch)
        
        
    writer.close()
    return model
   
class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, poison_indices, trigger, trigger_label):
        self.dataset = original_dataset
        self.poison_indices = poison_indices
        self.trigger = trigger
        self.trigger_label = trigger_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if idx in self.poison_indices:
            img = apply_trigger(img,self.trigger,"bottom-right")
            label = self.trigger_label
            return img, label
        else:
            return img, label