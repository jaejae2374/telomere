import time
import datetime
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_large
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

hyper_param_batch = 32
num_classes = 4

random_seed = 100
random.seed(random_seed)
torch.manual_seed(random_seed)

num_classes = 4

train_name = 'model5'

PATH = './scalp_weights/'

data_train_path = "./data/유형별 두피 이미지/Training/탈모"
data_validation_path = "./data/유형별 두피 이미지/Validation/탈모"

model = mobilenet_v3_large(weights="IMAGENET1K_V1", pretrained=True)

model = model.to(device)

transforms_train = transforms.Compose([
    transforms.Resize([int(600), int(600)], interpolation=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.9, saturation=0.9),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Lambda(lambda x: x.rotate(90)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_val = transforms.Compose([
    transforms.Resize([int(600), int(600)], interpolation=4),
    transforms.ColorJitter(contrast=0.9, saturation=0.9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_data_set = datasets.ImageFolder(data_train_path, transform=transforms_train)
val_data_set = datasets.ImageFolder(data_validation_path, transform=transforms_val)

dataloaders, batch_num = {}, {}

dataloaders['train'] = DataLoader(
    train_data_set,
    batch_size=hyper_param_batch,
    shuffle=True,
    num_workers=4
)
dataloaders['val'] = DataLoader(
    val_data_set,
    batch_size=hyper_param_batch,
    shuffle=False,
    num_workers=4
)

batch_num['train'], batch_num['val'] = len(train_data_set), len(val_data_set)

print('batch_size : %d,  train/val : %d / %d' % (hyper_param_batch, batch_num['train'], batch_num['val']))

class_names = train_data_set.classes
print(class_names)

def rand_bbox(size, lam):
    """이미지를 랜덤한 위치와 크기로 사각형으로 자른 후 좌표를 리턴합니다."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    beta = 1.0
    cutmix_prob = 0.5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    r = np.random.rand(1)
                    if beta > int(0.0) and r < cutmix_prob:
                        # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        # compute output
                        outputs = model(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                    else:
                        # compute output
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                
            epoch_end = time.time() - epoch_start
            
            print('Training epochs {} in {:.0f}m {:.0f}s'.format(epoch, epoch_end // 60, epoch_end % 60))
            print()
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    model.load_state_dict(best_model_wts)
        
    torch.save(model, PATH + 'aram_'+train_name+'.pt')
    torch.save(model.state_dict(), PATH + 'president_aram_'+train_name+'.pt')
    print('model saved')

    end_sec = time.time() - start_time
    end_times = str(datetime.timedelta(seconds=end_sec)).split('.')
    end_time = end_times[0]
    print("end time :", end_time)
    
    return model, best_idx, best_acc, train_loss, train_acc, val_loss, val_acc

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model.parameters(),lr = 1e-4)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

num_epochs = 200
train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
