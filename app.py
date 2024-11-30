import base64
import json
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

# Retrieve API configuration from environment variables
AMALIAI_BASE_URL = os.getenv("AMALIAI_BASE_URL")
AMALIAI_API_KEY = os.getenv("AMALIAI_API_KEY")
DEFAULT_MODEL_ID = os.getenv("AMALIAI_DEFAULT_MODEL_ID", "")


# Initialize session state for conversation track
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""

if "parent_message_id" not in st.session_state:
    st.session_state.parent_message_id = None


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


def send_amaliai_request(
    base_url,
    api_key,
    prompt,
    conversation_id,
    parent_message_id=None,
    model_id=None,
    stream=False,
):
    """
    Function to send request to AmaliAI.

    """
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "*/*",
    }

    # Construct payload
    payload = {"prompt": prompt, "stream": stream, "conversationId": conversation_id}

    # Add parent message ID if available
    if parent_message_id:
        payload["parentMessageId"] = parent_message_id

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


def display_conversation_history():
    """
    Display conversation history in the sidebar with improved tracking
    """
    st.sidebar.header("💬 Conversation History")

    # Allow clearing conversation history
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.conversation_history = ""
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.parent_message_id = None

    # Display conversation history
    if not st.session_state.conversation_history:
        st.sidebar.info("No conversation history yet.")
    else:
        messages = st.session_state.conversation_history.split("|")
        for idx, message in enumerate(reversed(messages)):
            role, content = message.split(":", 1)
            bg_color = "#242323" if idx % 2 == 0 else "#363333"
            st.sidebar.markdown(
                f"""
                <div style='background-color:{bg_color}; padding:10px; margin-bottom:5px; border-radius:5px'>
                <strong>{"You" if role == 'user' else "AmaliAI"}:</strong><br>
                {content}
                </div>
                """,
                unsafe_allow_html=True,
            )

def add_custom_styling():
    st.markdown("""
        <style>
        /* Main background and text colors */
        .stApp {
            background-color: #f7f9f4;  /* Light sage background */
        }

        /* Headers */
        h1, h2, h3 {
            color: #2c5530 !important;  /* Dark green */
        }

        /* Sidebar styling */
        .css-1d391kg {  /* Sidebar */
            background-color: #e8eed6 !important;  /* Light olive */
        }

        /* Buttons */
        .stButton>button {
            background-color: #678d58 !important;  /* Forest green */
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
        }

        .stButton>button:hover {
            background-color: #4a6b3d !important;  /* Darker green on hover */
        }

        /* File uploader */
        .stFileUploader {
            background-color: #ffffff !important;
            border: 2px dashed #678d58 !important;
            border-radius: 5px !important;
            padding: 1rem !important;
        }

        /* Chat messages and conversation styling */
        .stChatMessage {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
            border-left: 4px solid #678d58 !important;
        }

        /* Chat message text */
        .stChatMessage p {
            color: #2c5530 !important;  /* Dark green text */
            font-weight: 400 !important;
            opacity: 1 !important;
        }

        /* Message from user */
        .stChatMessageContent--user {
            background-color: #e8eed6 !important;  /* Light olive */
        }

        /* Message from assistant */
        .stChatMessageContent--assistant {
            background-color: #ffffff !important;
        }

        /* Text input fields */
        .stTextInput>div>div>input {
            background-color: #ffffff !important;
            border: 1px solid #678d58 !important;
            border-radius: 5px !important;
            color: #2c5530 !important;  /* Dark green text */
            font-weight: 500 !important;  /* Medium weight for better visibility */
            caret-color: #678d58 !important;  /* Cursor color */
        }

        /* Text input placeholder */
        .stTextInput>div>div>input::placeholder {
            color: #678d58 !important;  /* Forest green for placeholder */
            opacity: 0.7 !important;
        }

        /* Blinking cursor */
        .stTextInput>div>div>input:focus {
            caret-color: #2c5530 !important;  /* Dark green cursor */
            border-color: #2c5530 !important;  /* Dark green border on focus */
        }

        /* Dataframe styling */
        .dataframe {
            background-color: #ffffff !important;
            border: 1px solid #678d58 !important;
            border-radius: 5px !important;
        }

        /* Custom container for detected objects */
        .detected-objects {
            background-color: #ffffff !important;
            border: 1px solid #678d58 !important;
            border-radius: 5px !important;
            padding: 1rem !important;
            margin: 1rem 0 !important;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #678d58 !important;
        }

        /* Conversation text in the main chat area */
        .stMarkdown {
            color: #2c5530 !important;
            opacity: 1 !important;
        }

        /* Ensure all text in chat is visible */
        p, span, div {
            color: #2c5530 !important;
        }
        </style>
    """, unsafe_allow_html=True)

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
        page_title="🍅🍆Agriculture Chatbot",
        page_icon="🌽",
        initial_sidebar_state="expanded",
    )

    # Add the custom styling
    add_custom_styling()

    st.title("🌽 DevAI Crop Disease Detection and Prevention with AmaliAI")

    # Validate environment configuration
    if not AMALIAI_BASE_URL or not AMALIAI_API_KEY:
        st.error(
            "❌ Missing AmaliAI configuration. Please set AMALIAI_BASE_URL and AMALIAI_API_KEY in .env file."
        )
        return

    # Display conversation history in sidebar
    display_conversation_history()

    # Optional sidebar for advanced settings (collapsed by default)
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        model_id = st.text_input(
            "Model ID", value=DEFAULT_MODEL_ID, help="Leave blank to use default model"
        )
        stream_response = st.checkbox("Stream Response", value=False)

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
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Perform object detection
        original_image, detected_objects = detect_objects(uploaded_file)

        # Display original image with detections
        st.subheader("Image with Detected Objects")
        detected_image = visualize_detections(original_image, detected_objects)
        st.image(detected_image, channels="RGB")

        # Display detected objects
        st.subheader("Detected Objects")
        objects_df = [
            {"Class": obj["class"], "Confidence": f"{obj['confidence']:.2%}"}
            for obj in detected_objects
        ]
        st.dataframe(objects_df)

        # Question input for the image
        st.markdown("### Chat with AMALIAI")
        question = st.text_input(
            "### Chat with AmaliAI",
            placeholder="What objects are in this image?",
        )

        if question or st.session_state.conversation_history:
            st.write("---")

        # Prepare context about detected objects
        # objects_context = (
        #     "\n".join(
        #         [
        #             f"- {obj['class']} (confidence: {obj['confidence']:.2%})"
        #             for obj in detected_objects
        #         ]
        #     )
        #     if detected_objects
        #     else "No Objects detected"
        # )

        # Process question if provided
        if uploaded_file and question:
            # Convert image to base64 for potential use
            img_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Construct full prompt
            full_prompt = f"""Question: {question}"""

            # Send request to AmaliAI
            try:
                # Append the question to conversation history
                if st.session_state.conversation_history:
                    st.session_state.conversation_history += f"|user:{question}"
                else:
                    st.session_state.conversation_history = f"user:{question}"

                # includes all user and ai convo
                full_convo = st.session_state.conversation_history
                print("full convo: ", full_convo)

                response = send_amaliai_request(
                    base_url=AMALIAI_BASE_URL,
                    api_key=AMALIAI_API_KEY,
                    prompt=full_convo,
                    conversation_id=st.session_state.conversation_id,
                    parent_message_id=st.session_state.parent_message_id,
                    model_id=model_id or None,
                    stream=stream_response,
                )

                # Addd assistant response to conversation History
                # st.session_state.conversation_history.append(
                #     {"role": "assistant", "content": response}
                # )

                # Append the response to conversation history
                st.session_state.conversation_history += f"|assistant:{response}"
            

                # Update parent message ID for context tracking
                st.session_state.parent_message_id = str(uuid.uuid4())

                # add_to_conversation_history("assistant", response)
                #
                # st.subheader("AmaliAI's Response")
                # st.write(response)

            except Exception as e:
                st.error(f"An error occured: {str(e)}")

        # Display the conversation history on the main page
        for message in st.session_state.conversation_history.split("|"):
            if ":" not in message:
                continue  # Skip invalid messages
            role, content = message.split(":", 1)
            if role == "user":
                st.chat_message("user").markdown(content)
            else:
                st.chat_message("assistant").markdown(content)


if __name__ == "__main__":
    main()
