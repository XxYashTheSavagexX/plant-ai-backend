import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import bcrypt

app = Flask(__name__)

UPLOAD_FOLDER = "images"
MODEL_PATH = "plant_model.pth"
CLASSES_PATH = "classes.json"
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD CLASSES
# -----------------------------

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

num_classes = 50

# -----------------------------
# LOAD MODEL (ONCE)
# -----------------------------

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 50)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# JSON HELPERS
# -----------------------------

def load_json(path):

    if not os.path.exists(path):
        return {}

    try:
        with open(path,"r") as f:
            return json.load(f)
    except:
        return {}

def save_json(path,data):

    with open(path,"w") as f:
        json.dump(data,f,indent=2)

# -----------------------------
# AUTH ROUTES
# -----------------------------

@app.route("/register", methods=["POST"])
def register():

    data = request.json

    email = data.get("email")
    password = data.get("password")

    users = load_json(USERS_FILE)

    if email in users:
        return jsonify({"error":"User exists"}),400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    users[email] = {
        "password": hashed
    }

    save_json(USERS_FILE,users)

    return jsonify({"success":True})


@app.route("/login", methods=["POST"])
def login():

    data = request.json

    email = data.get("email")
    password = data.get("password")

    users = load_json(USERS_FILE)

    if email not in users:
        return jsonify({"error":"Invalid login"}),401

    stored = users[email]

if isinstance(stored, dict):
    stored = stored["password"]

    if not bcrypt.checkpw(password.encode(), stored.encode()):
        return jsonify({"error":"Invalid login"}),401

    return jsonify({
        "email": email
    })

# -----------------------------
# HISTORY FUNCTIONS
# -----------------------------

def add_history(email,item):

    history = load_json(HISTORY_FILE)

    if email not in history:
        history[email] = []

    history[email].insert(0,item)

    save_json(HISTORY_FILE,history)


def get_history(email):

    history = load_json(HISTORY_FILE)

    if email not in history:
        return []

    return history[email]

# -----------------------------
# PREDICT
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error":"no image"}),400

    file = request.files["image"]
    email = request.form.get("email")

    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(path)

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image)
        probs = torch.softmax(output,1)

        conf, pred = torch.max(probs,1)

        label = classes[pred.item()]

        health = int(conf.item()*100)

    plant = label.split("___")[0]
    disease = label.split("___")[1]

    result = {
        "plant": plant,
        "disease": disease,
        "health": health,
        "image": filename,
        "paragraph": "The plant shows signs related to this disease classification.",
        "issues": [
            "Leaf discoloration",
            "Possible fungal infection"
        ],
        "tips": [
            "Remove infected leaves",
            "Avoid overhead watering",
            "Improve air circulation"
        ],
        "water": "Medium",
        "sun": "High",
        "soil": "Moist",
        "highlight": filename
    }

    if email:
        add_history(email,result)

    return jsonify(result)

# -----------------------------
# HISTORY ROUTE
# -----------------------------

@app.route("/history/<email>", methods=["GET"])
def history(email):

    user_history = get_history(email)

    return jsonify(user_history)

# -----------------------------
# SERVE IMAGES
# -----------------------------

@app.route("/images/<path:filename>")
def images(filename):

    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    print("Plant AI Server Running...")

    app.run(host="0.0.0.0", port=port)
