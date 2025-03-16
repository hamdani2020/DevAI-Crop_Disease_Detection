import base64
import os
from io import BytesIO

import cv2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# Load environment variables
load_dotenv()


# Load YOLO model
def load_yolo_model():
    return YOLO("model.pt")


# Perform object detection
def detect_objects(image, confidence_threshold=0.5):
    model = load_yolo_model()
    img = Image.open(image)
    img_array = np.array(img)
    results = model(img_array)

    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects.append(
                {"class": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
            )
    return img_array, detected_objects


# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Flask app
app = Flask(__name__)
CORS(app)


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    confidence_threshold = float(request.form.get("confidence_threshold", 0.5))

    # Perform object detection
    original_image, detected_objects = detect_objects(file, confidence_threshold)

    # Convert image to base64 for response
    img_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"image": img_base64, "detected_objects": detected_objects})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    print("Received data:", data)  # Debugging log

    image_base64 = data.get("image")
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Construct full prompt
    objects_context = data.get("objects_context", "No objects detected")
    full_prompt = f"""I have uploaded an image with the following detected objects: {objects_context} Question: {question}"""
    print("Full prompt:", full_prompt)  # Debugging log

    # Send request to Gemini API
    try:
        response = model.generate_content(full_prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        print("Gemini API error:", str(e))  # Debugging log
        return jsonify({"error": "Failed to get response from Gemini API"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
