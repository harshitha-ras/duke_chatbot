#!/bin/bash
# /home/harsh/duke_chatbot/deploy.sh

cd /home/harsh/duke_chatbot
git pull origin main

# Kill existing processes
pkill -f "flask --app backend_app.py" || true
pkill -f "streamlit run streamlit_app.py" || true

# Activate virtual environment
source /home/harsh/venv/bin/activate

# Start Flask backend
nohup flask --app backend_app.py run --host=127.0.0.1 --port=5000 > flask.log 2>&1 &

# Wait for Flask to start
sleep 5

# Start Streamlit frontend
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

echo "Deployment completed at $(date)"
