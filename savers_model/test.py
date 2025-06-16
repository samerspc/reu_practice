import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import json
import matplotlib.pyplot as plt

# === Настройки ===
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print("✅ Using device:", device)

model_path = "food101_small_model.pt"
idx_to_class_path = "idx_to_class_small.json"
image_path = "/content/dataset/images/baklava/1021344.jpg"  # Поменяй на свой файл

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Загрузка словаря классов ===
with open(idx_to_class_path, "r") as f:
    idx_to_class = json.load(f)

num_classes = len(idx_to_class)

# === Загрузка модели ===
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Предсказание по изображению ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_index = outputs.max(1)
        predicted_class = idx_to_class[str(pred_index.item())]

    # Показ результата
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"🍽 Предсказано: {predicted_class}")
    plt.show()

# === Запуск
predict_image(image_path)