import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import urllib.request
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_URL = "https://huggingface.co/NazarBai/mushroom-resnet50/resolve/main/model.pt"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "model.pt")

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # ✅ use MODEL_PATH
model.eval()

class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma',
               'Hygrocybe', 'Lactarius', 'Mushrooms', 'Russula', 'Suillus']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Mushroom Classifier")
file = st.file_uploader("Upload a mushroom image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Running prediction..."):
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output[0], dim=0)

    top_probs, top_idxs = probs.topk(3)
    st.subheader("Top Predictions:")
    for i in range(3):
        st.write(f"{class_names[top_idxs[i]]}: {top_probs[i]*100:.2f}%")


