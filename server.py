import os
import json
import uuid
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import bcrypt
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "images"
MODEL_PATH = "plant_model.pth"
CLASSES_PATH = "classes.json"
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

NUM_CLASSES = 50

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

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
# AUTH
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

    users[email] = {"password": hashed}

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

    return jsonify({"email":email})

# -----------------------------
# HISTORY
# -----------------------------

def add_history(item):

    history = load_json(HISTORY_FILE)

    if not isinstance(history, list):
        history = []

    history.insert(0, item)

    save_json(HISTORY_FILE, history)


def get_history():

    history = load_json(HISTORY_FILE)

    if not isinstance(history, list):
        return []

    return history

# -----------------------------
# PREDICT
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({"error":"no image"}),400

        file = request.files["image"]
        email = request.form.get("email")

        print("EMAIL RECEIVED:", email)

        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        # -----------------------------
        # LOAD IMAGE SAFELY
        # -----------------------------

        try:
            image = Image.open(path).convert("RGB")
        except:
            return jsonify({"error":"invalid image"}),400

        image_tensor = transform(image).unsqueeze(0)

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------

        with torch.no_grad():

            output = model(image_tensor)
            probs = torch.softmax(output,1)

            conf, pred = torch.max(probs,1)

            idx = pred.item()

            if idx >= NUM_CLASSES:
                idx = NUM_CLASSES - 1

            if idx >= len(classes):
                idx = idx % len(classes)

            label = classes[idx]
            health = int(conf.item()*100)

        plant = label.split("___")[0]
        disease = label.split("___")[1]

        # -----------------------------
        # DAMAGE DETECTION
        # -----------------------------

        img = cv2.imread(path)

        if img is None:
            return jsonify({"error":"image processing failed"}),400

        img = cv2.resize(img,(1024,1024))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([15,80,80]),
            np.array([35,255,255])
        )

        brown_mask = cv2.inRange(
            hsv,
            np.array([5,50,50]),
            np.array([20,255,200])
        )

        white_mask = cv2.inRange(
            hsv,
            np.array([0,0,200]),
            np.array([180,60,255])
        )

        mask = yellow_mask | brown_mask | white_mask

        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        highlight = img.copy()

        for c in contours:

            area = cv2.contourArea(c)

            if area > 300:

                x,y,w,h = cv2.boundingRect(c)

                cv2.rectangle(
                    highlight,
                    (x,y),
                    (x+w,y+h),
                    (0,0,255),
                    3
                )

        highlight_name = f"highlight_{filename}"
        highlight_path = os.path.join(UPLOAD_FOLDER, highlight_name)

        cv2.imwrite(highlight_path, highlight)

        # -----------------------------
        # RESPONSE
        # -----------------------------

        result = {
            "plant": plant,
            "disease": disease,
            "health": health,
            "image": filename,
            "highlight": highlight_name,
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
            "soil": "Moist"
        }

        add_history(result)

        return jsonify(result)

    except Exception as e:

        print("SERVER ERROR:", e)

        return jsonify({"error":"server error"}),500

# -----------------------------
# HISTORY ROUTE
# -----------------------------

@app.route("/history")
def history():

    return jsonify(get_history())

# -----------------------------
# IMAGE SERVER
# -----------------------------

@app.route("/images/<path:filename>")
def images(filename):

    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    print("Plant AI Server Running...")

    app.run(host="0.0.0.0", port=port)
