from flask import Flask, request, jsonify
import random # Replace with your actual chatbot logic
from agent import create_duke_agent, process_user_query

app = Flask(__name__)

# --- Replace this with your actual chatbot logic ---
def get_chatbot_response(user_message):
    # Example: Simple echo bot or predefined responses
    responses = [
        f"You said: {user_message}",
        "That's interesting!",
        "Tell me more."
    ]
    # Integrate your model loading and prediction here
    return random.choice(responses)
# --- End of chatbot logic section ---

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        bot_response = get_chatbot_response(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Listen on 0.0.0.0 to be accessible externally *if needed*,
    # but Streamlit will call it locally via 127.0.0.1
    app.run(host='127.0.0.1', port=5000, debug=False) # Run on localhost only initially

