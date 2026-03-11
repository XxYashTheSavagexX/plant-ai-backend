from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import io
import os
import json
import datetime
import numpy as np
import cv2
import bcrypt

app = Flask(__name__)

# ---------------- CONFIG ----------------

IMAGE_SIZE = 128
MODEL_PATH = "plant_model.pth"
HISTORY_FILE = "history.json"
USERS_FILE = "users.json"

if not os.path.exists("images"):
    os.makedirs("images")

# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ---------------- LOAD CLASSES ----------------

with open("classes.json") as f:
    classes = json.load(f)

# ---------------- LOAD MODEL (ON STARTUP) ----------------

print("Loading AI model...")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

num_classes = checkpoint["classifier.1.weight"].shape[0]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model.load_state_dict(checkpoint)
model.eval()

print("AI model loaded successfully")

# ---------------- USER SYSTEM ----------------

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@app.route("/register", methods=["POST"])
def register():

    data = request.json
    email = data["email"]
    password = data["password"]

    users = load_users()

    if email in users:
        return jsonify({"error":"User exists"}),400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    users[email] = hashed
    save_users(users)

    return jsonify({"message":"Account created"})

@app.route("/login", methods=["POST"])
def login():

    data = request.json
    email = data["email"]
    password = data["password"]

    users = load_users()

    if email not in users:
        return jsonify({"error":"Invalid login"}),401

    stored_hash = users[email]

    if bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return jsonify({"email":email})

    return jsonify({"error":"Invalid login"}),401

# ---------------- HISTORY ----------------

def save_history(entry):

    try:
        with open(HISTORY_FILE,"r") as f:
            data=json.load(f)
    except:
        data=[]

    data.append(entry)

    with open(HISTORY_FILE,"w") as f:
        json.dump(data,f)

@app.route("/history", methods=["GET"])
def history():

    try:
        with open(HISTORY_FILE,"r") as f:
            data=json.load(f)
    except:
        data=[]

    return jsonify(data)

# ---------------- IMAGE PROCESS ----------------

def highlight_disease(pil_image):

    img=np.array(pil_image)

    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    lower=np.array([10,80,80])
    upper=np.array([40,255,255])

    mask=cv2.inRange(hsv,lower,upper)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    output=img.copy()

    for c in contours:

        if cv2.contourArea(c)<200:
            continue

        x,y,w,h=cv2.boundingRect(c)

        cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),3)

    return output

# ---------------- CARE TEXT ----------------

def generate_texts(name):

    if "healthy" in name.lower():
        return "Healthy","The plant appears healthy.",[],[]

    return (
        "Possible Disease",
        "The plant shows signs of disease or stress.",
        ["Leaf discoloration","Possible infection"],
        ["Remove damaged leaves","Avoid overwatering","Provide sunlight"]
    )

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    file=request.files["image"]

    pil_image=Image.open(io.BytesIO(file.read())).convert("RGB")

    filename=f"{datetime.datetime.now().timestamp()}.jpg"

    pil_image.save(f"images/{filename}")

    highlighted=highlight_disease(pil_image)

    highlight_name=f"highlight_{filename}"

    cv2.imwrite(f"images/{highlight_name}",highlighted)

    image=transform(pil_image).unsqueeze(0)

    with torch.no_grad():

        outputs=model(image)

        probs=torch.nn.functional.softmax(outputs,dim=1)

        confidence,predicted=torch.max(probs,1)

    raw_name=classes[predicted.item()]

    plant_name=raw_name.replace("___"," - ").replace("_"," ")

    health=95 if "healthy" in plant_name.lower() else 60

    disease,paragraph,issues,tips=generate_texts(plant_name)

    result={
        "plant":plant_name,
        "disease":disease,
        "health":health,
        "paragraph":paragraph,
        "issues":issues,
        "tips":tips,
        "image":filename,
        "highlight":highlight_name,
        "date":str(datetime.datetime.now())
    }

    save_history(result)

    return jsonify(result)

# ---------------- IMAGE ROUTE ----------------

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("images",filename)

# ---------------- RUN ----------------

if __name__=="__main__":

    print("Plant AI Server Running...")

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
