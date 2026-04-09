import torch
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
).to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

img = Image.open("Spectrograms/noise/silence262.png").convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    embedding = model(img)

print(embedding)