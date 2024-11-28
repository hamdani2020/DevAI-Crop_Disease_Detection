import requests
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


x_api_key = os.getenv("X-API-KEY")

X_API_KEY = os.getenv("X-API-KEY")



# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI")  # Replace with your MongoDB URI
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

st.title("🌽 Agriculture Chatbot")
st.caption("🚜 A Streamlit chatbot powered by a public API and MongoDB")

# Initialize session state for conversation
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = "Assistant: How can I assist you today?"

# Load conversation from MongoDB
def load_conversation(conversation_id):
    if conversation_id:
        conversation = collection.find_one({"conversation_id": conversation_id})
        if conversation:
            return conversation["conversation_history"]
    return "Assistant: How can I assist you today?"

# Save conversation to MongoDB
def save_conversation(conversation_id, conversation_history):
    if not conversation_id:
        # Insert a new conversation
        result = collection.insert_one({"conversation_id": st.session_state["conversation_id"], 
                                        "conversation_history": conversation_history})
        st.session_state["conversation_id"] = result.inserted_id
    else:
        # Update an existing conversation
        collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"conversation_history": conversation_history}},
            upsert=True
        )

# Display chat history
chat_history = st.session_state["conversation_history"]
for line in chat_history.split("|"):
    role, message = line.split(":", 1)
    st.chat_message(role.strip().lower()).write(message.strip())

# Handle new user input
if prompt := st.chat_input():
    # if not public_api_key:
    #     st.info("Please add your Public API key to continue.")
    #     st.stop()

    # Append user input to conversation history
    st.session_state["conversation_history"] += f"|User: {prompt}"
    st.chat_message("user").write(prompt)
    new_prompt = st.session_state["conversation_history"]

    print("Convo_Id",st.session_state["conversation_id"]) #hereeeeeeeeeeeeeeeeee
    # Send prompt to the public API

    url =  os.getenv("AI_URI") # Replace with your API endpoint
    headers = {
               "X-Api-Key": x_api_key
    }

    url = os.getenv("AI_URI")  # Replace with your API endpoint
    headers = {
               "X-Api-Key": X_API_KEY

               }
    payload = {
        # "conversationId": st.session_state["conversation_id"],
        "prompt": prompt,
        "stream": False
        # "history": st.session_state["conversation_history"]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        if response_data.get("success"):
            api_response = response_data["data"]["content"]
            st.session_state["conversation_id"] = response_data["data"]["conversationId"]

            # Append assistant response to conversation history
            st.session_state["conversation_history"] += f"|Assistant: {api_response}"
            st.chat_message("assistant").write(api_response)

            print("Conversation ID:", st.session_state.get("conversation_id"))
            print("Conversation History:", st.session_state.get("conversation_history"))


            # Save the updated conversation to MongoDB
            try:
                save_conversation(st.session_state["conversation_id"], st.session_state["conversation_history"])
                print("Conversation saved successfully.")
            except Exception as e:
                print("Error saving conversation:", str(e))

        else:
            st.error(f"API Error: {response_data.get('status', 'Unknown Error')}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
