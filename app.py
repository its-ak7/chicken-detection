from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return open("index.html").read()

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    results = model(img)[0]

    healthy = 0
    sick = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        if cls == 0:
            label = "Healthy"
            color = (0,255,0)
            healthy += 1
        else:
            label = "Sick"
            color = (0,0,255)
            sick += 1

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(img,label,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, img)

    return jsonify({
        "healthy": healthy,
        "sick": sick,
        "image": f"/output/{filename}"
    })

@app.route("/output/<filename>")
def get_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename),
                     mimetype='image/jpeg')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
