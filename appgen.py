import base64
import os
import uuid
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

APP_VERSION = "0.0.1"

# Retrieve API configuration from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# AMALIAI_API_KEY = os.getenv("AMALIAI_API_KEY")
# DEFAULT_MODEL_ID = os.getenv("AMALIAI_DEFAULT_MODEL_ID", "")


# Initialize session state for conversation track
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "parent_message_id" not in st.session_state:
    st.session_state.parent_message_id = None


# Load YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("model.pt")  # You can change to a different pre-trained model if needed


# Perform object detection
def detect_objects(image, confidence_threshold=0.5):
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


def send_amaliai_request(
    api_key,
    prompt,
    image_base64=None,
    model_name="gemini-2.0-flash",
    stream=True,
):
    """
    Function to send request to AmaliAI.

    """

    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
    }

    # Construct payload
    payload = {"contents": []}

    # Add text prompt
    if image_base64:
        image_part = {
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": image_base64}},
            ]
        }
        payload["contents"].append(image_part)
    else:
        payload["content"].append({"parts": [{"text": prompt}]})

    try:
        url = f"{base_url}?key={api_key}"

        response = requests.post(url, headers=headers, json=payload)

        response.raise_for_status()

        response_json = response.json()

        if "candidates" in response_json and response_json["candidates"]:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "No response generated"

    except requests.RequestException as e:
        return f"Request failed: {str(e)}"


def display_conversation_history():
    """
    Display conversation history in the sidebar with improved tracking
    """
    st.sidebar.header("💬 Conversation History")

    # Allow clearing conversation history
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.parent_message_id = None

    # Display conversation history
    if not st.session_state.conversation_history:
        st.sidebar.info("No conversation history yet.")

    else:
        # Reverse the history to show most recent
        num_summary_messages = 3
        summary_messages = st.session_state.conversation_history[-num_summary_messages:]

        for idx, message in enumerate(summary_messages, 1):
            content = (
                message["content"][:100] + "..."
                if len(message["content"]) > 100
                else message["content"]
            )

            st.sidebar.markdown(
                f"""
                <div style='background-color:#1a4203; padding:8px; margin-bottom:3px; border-radius:5px'>
                <strong>{"You" if message['role'] == 'user' else "GreenAI"}:</strong><br>
                {content}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Add a total message count
        total_messages = len(st.session_state.conversation_history)
        st.sidebar.markdown(f"**Total Messages:** {total_messages}")


# def add_to_conversation_history(role, content):
#     """
#     Add a message to the conversation history
#
#     Parameters:
#         role (str): Role of the message sender ('user', or 'assistant')
#         content (str): Content of the message
#     """
#     st.session_state.conversation_history.append({"role": role, "content": content})


# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="🍅🍆GreenAI",
        page_icon="🌽",
        initial_sidebar_state="expanded",
    )

    st.markdown("<h1 style='color: green;'>🌽 GreenAI🍀</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='color: green;'>DevAI Crop Disease Detection and Prevention</h4>",
        unsafe_allow_html=True,
    )

    # Validate environment configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("Missing Gemini API key. Please set GEMINI_API_KEY in .env file")

    # Display conversation history in sidebar
    display_conversation_history()

    # Optional sidebar for advanced settings (collapsed by default)
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        stream_response = st.checkbox("Stream Response", value=False)

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Set minimum confidence level for object detection. Higher values show only more certain detections",
        )

        # Display chat conversation on the main page
        # st.header(" Chat with AmaliAI")
        #
        # chat_container = st.container()
        # with chat_container:
        #     for message in st.session_state.conversation_history:
        #         if message["role"] == "user":
        #             st.chat_message("user").markdown(message["content"])
        #         else:
        #             st.chat_message("assistant").markdown(message["content"])

    # File uploader for images
    uploaded_file = st.file_uploader("AgroDetect", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.markdown(
        "*Developed with 💚 for Agricultural Innovation and Happy Farmer's Day Celebration*"
    )

    if uploaded_file is not None:
        # Perform object detection
        original_image, detected_objects = detect_objects(uploaded_file)

        # Display original image with detections
        st.subheader("Image with Detected Objects")
        detected_image = visualize_detections(original_image, detected_objects)
        st.image(detected_image, channels="RGB")

        # Display detected objects
        st.subheader("Detected Objects")
        if detected_objects:
            objects_df = [
                {"Class": obj["class"], "Confidence": f"{obj['confidence']:.2%}"}
                for obj in detected_objects
            ]
            st.dataframe(objects_df)
        else:
            st.info(
                f"No objects detected abot the {confidence_threshold:.2%} confidence threshold"
            )
        # objects_df = [
        #     {"Class": obj["class"], "Confidence": f"{obj['confidence']:.2%}"}
        #     for obj in detected_objects
        # ]
        # st.dataframe(objects_df)

        # Question input for the image
        st.markdown("### Chat with GreenAI")

        conversation_container = st.container()

        question = st.chat_input(
            placeholder="What objects are in this image?",
        )

        with conversation_container:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.chat_message("user").markdown(message["content"])
                else:
                    st.chat_message("assistant").markdown(message["content"])

        # Prepare context about detected objects
        objects_context = (
            "\n".join(
                [
                    f"- {obj['class']} (confidence: {obj['confidence']:.2%})"
                    for obj in detected_objects
                ]
            )
            if detected_objects
            else "No Objects detected"
        )

        # Process question if provided
        if uploaded_file and question:
            # Convert image to base64 for potential use
            img_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct full prompt
            full_prompt = f"""I have uploaded an image eith the following detected objects: {objects_context} Question: {question}"""

            # Send request to AmaliAI
            try:
                st.session_state.conversation_history.append(
                    {"role": "user", "content": question}
                )

                response = send_amaliai_request(
                    api_key=GEMINI_API_KEY,
                    prompt=full_prompt,
                    image_base64=img_base64,
                    stream=stream_response,
                )

                # Addd assistant response to conversation History
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response}
                )

                # Update parent message ID for context tracking
                # st.session_state.parent_message_id = str(uuid.uuid4())

                # add_to_conversation_history("assistant", response)
                #
                # st.subheader("AmaliAI's Response")
                # st.write(response)

                st.rerun()

            except Exception as e:
                st.error(f"An error occured: {str(e)}")

        # for message in st.session_state.conversation_history:
        #     if message["role"] == "user":
        #         st.chat_message("user").markdown(message["content"])
        #     else:
        #         st.chat_message("assistant").markdown(message["content"])


if __name__ == "__main__":
    main()

