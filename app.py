from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model import load_model

app = Flask(__name__)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load classes
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

num_classes = len(class_names)

# Load model
model = load_model("best_model.pth", num_classes, device)

# TRANSFORM — EĞİTİMLE UYUMLU
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    top5 = torch.topk(probs, 5)

    results = []
    for idx, score in zip(top5.indices, top5.values):
        results.append({
            "class": class_names[idx],
            "confidence": float(score)
        })

    return jsonify(results)

@app.route("/status")
def status():
    return jsonify({
        "device": device,
        "checkpoint": "best_model.pth",
        "classes": num_classes,
        "status": "ready"
    })

if __name__ == "__main__":
    app.run(debug=True)
