import base64
import json
import os
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Retrieve API configuration from environment variables
AMALIAI_BASE_URL = os.getenv("AMALIAI_BASE_URL")
AMALIAI_API_KEY = os.getenv("AMALIAI_API_KEY")
DEFAULT_MODEL_ID = os.getenv("AMALIAI_DEFAULT_MODEL_ID", "")


# Load YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("model.pt")  # You can change to a different pre-trained model if needed


# Perform object detection
def detect_objects(image):
    model = load_yolo_model()

    # Convert Streamlit uploaded file to OpenCV format
    img = Image.open(image)
    img_array = np.array(img)

    # Perform detection
    results = model(img_array)

    # Extract detected objects and their details
    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Get confidence and bounding box
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_objects.append(
                {"class": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
            )

    return img_array, detected_objects


# Visualize detected objects
def visualize_detections(image, detections):
    img_with_boxes = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"

        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return img_with_boxes


# Function to send request to AmaliAI
def send_amaliai_request(base_url, api_key, prompt, model_id=None, stream=False):
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "*/*",
    }

    # Construct payload
    payload = {"prompt": prompt, "stream": stream}

    # Add model_id if provided
    if model_id:
        payload["modelId"] = model_id

    try:
        response = requests.post(
            f"{base_url}/public/chat", headers=headers, data=json.dumps(payload)
        )

        # Handle response based on stream status
        if stream:
            # For streaming, you'd need to parse SSE events
            # This is a simplified version
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        try:
                            json_data = json.loads(decoded_line[5:])
                            if "content" in json_data:
                                full_response += json_data.get("content", "")
                        except json.JSONDecodeError:
                            pass
            return full_response
        else:
            # For non-streaming
            response_json = response.json()
            if response_json.get("success"):
                return response_json["data"]["content"]
            else:
                return f"Error: {response_json.get('error', 'Unknown error')}"

    except requests.RequestException as e:
        return f"Request failed: {str(e)}"


# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="🌽 Agriculture Chatbot",
        page_icon="🌽",
        initial_sidebar_state="collapsed",
    )

    st.title("🌽 DevAI Crop Disease Detection and Prevention")

    # Validate environment configuration
    if not AMALIAI_BASE_URL or not AMALIAI_API_KEY:
        st.error(
            "❌ Missing AmaliAI configuration. Please set AMALIAI_BASE_URL and AMALIAI_API_KEY in .env file."
        )
        return

    # Optional sidebar for advanced settings (collapsed by default)
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        model_id = st.text_input(
            "Model ID", value=DEFAULT_MODEL_ID, help="Leave blank to use default model"
        )
        stream_response = st.checkbox("Stream Response", value=False)

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Perform object detection
        original_image, detected_objects = detect_objects(uploaded_file)

        # Display original image with detections
        st.subheader("Image with Detected Objects")
        detected_image = visualize_detections(original_image, detected_objects)
        st.image(detected_image, channels="BGR")

        # Display detected objects
        st.subheader("Detected Objects")
        objects_df = [
            {"Class": obj["class"], "Confidence": f"{obj['confidence']:.2%}"}
            for obj in detected_objects
        ]
        st.dataframe(objects_df)

        # Question input for the image
        question = st.text_input(
            "Ask something about the image",
            placeholder="What objects are in this image?",
        )

        # Prepare context about detected objects
        objects_context = "\n".join(
            [
                f"- {obj['class']} (confidence: {obj['confidence']:.2%})"
                for obj in detected_objects
            ]
        )

        # Process question if provided
        if uploaded_file and question:
            # Convert image to base64 for potential use
            img_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct full prompt
            full_prompt = f"""Detected objects in the image:
{objects_context}

Image details are included. {question}"""

            # Send request to AmaliAI
            try:
                response = send_amaliai_request(
                    base_url=AMALIAI_BASE_URL,
                    api_key=AMALIAI_API_KEY,
                    prompt=full_prompt,
                    model_id=model_id or None,
                    stream=stream_response,
                )

                # Display response
                st.subheader("AmaliAI's Response")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
