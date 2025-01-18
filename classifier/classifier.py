import torch
from torchvision import transforms

def classify(image, model,class_names):
    transformation = transforms.Compose([
        transforms.Resize((60,60)),
        transforms.ToTensor()
    ])
    
    image = transformation(image).unsqueeze(0) ## (C,H,W) ==> (B,C,H,W)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image = image.to(device)
    model = model.to(device)
    
    prediction = model(image)
    
    score, index = torch.max(prediction,1)
    class_name = class_names[index.item()]
    confidence_score = int(score * 100)
    
    return class_name,confidence_score