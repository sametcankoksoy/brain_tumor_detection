import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("./brain_tumor_model.pth", map_location=device))
model.to(device)
model.eval()

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_names = ['Healthy (No)', 'Tumor Detected (Yes)']
    result = class_names[predicted.item()]
    score = confidence.item() * 100
    
    return result, score

test_image_path = "C0472761-Meningioma_brain_tumour,_MRI_scan.jpg"
prediction, confidence_score = predict_image(test_image_path)

print(f"Model Prediction: {prediction}")
print(f"Confidence Score: {confidence_score:.2f}%")