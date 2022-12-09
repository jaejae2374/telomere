from django.http import HttpResponse
from django.shortcuts import render
import torch
from torchvision import transforms, datasets
from telo.models import Scalp

LABLE_TAGS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
}
SCALP_STATUS = {
    '0': "양호",
    '1': "경증",
    '2': "중등도",
    '3': "중증",
}
MODEL_PATH = './'
DATA_PATH = './images'
    
def index(request):
    if request.method=='POST':
        image = request.FILES['chooseFile']
        scalp = Scalp.objects.create(name=str(image), image=image)
        result = diagnose()
        return render(request, 'result.html', {
            'result': result,
            'scalp': scalp
        })
    return render(request, 'index.html')

def diagnose():
    device = torch.device('cpu')
    model = torch.load(MODEL_PATH + 'aram_model5.pt', map_location=device)
    model.eval()
    transforms_val = transforms.Compose([
        transforms.Resize([int(600), int(600)], interpolation=4),
        transforms.ColorJitter(contrast=0.9, saturation=0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data_set = datasets.ImageFolder(DATA_PATH, transform=transforms_val)
    input_img = test_data_set[-1][0].unsqueeze(dim=0).to(device)
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = LABLE_TAGS[argmax.item()]
    return SCALP_STATUS[pred]






