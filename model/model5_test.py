import torch
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


train_name = 'model5'

PATH = './scalp_weights/'
data_test_path = "./data/유형별 두피 이미지/Test/탈모"

num_classes = 4

label_tags = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
}
    
hyper_param_batch = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nThreads = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transforms_val = transforms.Compose([
    transforms.Resize([int(600), int(600)], interpolation=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data_set = datasets.ImageFolder(data_test_path, transform=transforms_val)

test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=hyper_param_batch,
                                        shuffle=False, num_workers=nThreads)

# load model
model = torch.load(PATH + 'aram_'+train_name+'.pt')
model.load_state_dict(torch.load(PATH + 'president_aram_'+train_name+'.pt'), strict=False) 
model.eval()

# inference
columns = int(len(test_data_set) / num_classes)
rows = int(len(test_data_set) / columns)

acc = 0.0

fig = plt.figure(figsize=(30,30))
idx = 0
for i in range(1, len(test_data_set)+1):
    input_img = test_data_set[i-1][0].unsqueeze(dim=0).to(device) 
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = label_tags[argmax.item()]
    label = label_tags[test_data_set[i-1][1]]
    
    fig.add_subplot(rows, columns, i)
    if pred == label:
        plt.title(f"[Correct]\nPred: {pred}\nTrue: {label}", fontdict={'fontsize':5})
        cmap = 'Blues'
        acc += 1
    else:
        plt.title(f"[Wrong]\nPred: {pred}\nTrue: {label}", fontdict={'fontsize':5})
        cmap = 'Reds'
    plot_img = test_data_set[i-1][0][0,:,:]
    plt.imshow(plot_img, cmap=cmap)
    plt.axis('off')
