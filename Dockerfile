FROM python:3.10-slim

# Keep Python behavior predictable in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MODEL_URL=https://huggingface.co/dinesh-moorthy/bike-rental-model/resolve/main/xgb_bike_model.joblib

WORKDIR /app

# Install Python dependencies first for better build caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application source.
COPY . /app

EXPOSE 8501

# Streamlit default server settings for container runtime.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
