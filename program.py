from torchvision.models import resnet18
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F


model = resnet18()


model.fc = torch.nn.Linear(in_features=512, out_features=20)


state_dict = torch.load("trained_model3.pth", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)

model.eval()

image_dir = "./NewSpec"


files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


if not files:
    raise FileNotFoundError("No spectrograms found in the directory.")


most_recent_file = max(files, key=os.path.getmtime)



image = Image.open(most_recent_file)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)  


with torch.no_grad():
    output = model(input_tensor)  
    probabilities = F.softmax(output, dim=1)  



class_1 = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']
current_class = ""
class_1_ = 0

print("Probabilities for each class:")
for i, prob in enumerate(probabilities[0]):
    if i < 10:
        current_class = f"f{i+1}"
        print(f"{current_class}: {prob:.4f}")
        if (lambda x: x in class_1)(current_class):
            class_1_ = class_1_ + prob

    else:
        current_class = f"m{i-9}"
        print(f"{current_class}: {prob:.4f}")
        if (lambda x: x in class_1)(current_class):
            class_1_ = class_1_ + prob
print(f"Probability that the audio belongs to class 1 is: {class_1_:.4f}")

