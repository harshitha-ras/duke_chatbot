import streamlit as st
import requests

st.set_page_config(page_title="My Chatbot") # [6]
st.title("My Chatbot Interface") # [2, 6]

# URL for the backend API running on the *same* VM
BACKEND_URL = "http://127.0.0.1:5000/chat" # Calls the Flask app locally

# Initialize chat history in session state [2, 6, 8]
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history [2, 6, 8]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input [2]
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from backend API
    try:
        response = requests.post(BACKEND_URL, json={"message": prompt})
        response.raise_for_status() # Raise an exception for bad status codes
        bot_response = response.json().get("response", "Sorry, I encountered an error.")
    except requests.exceptions.RequestException as e:
        bot_response = f"Error contacting backend: {e}"
    except Exception as e:
         bot_response = f"An error occurred: {e}"


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
