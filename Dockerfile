FROM python:3.11-slim

# ffmpeg for H.264 export; libgl/libglib for the opencv-python that ultralytics pulls in
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the YOLO weights into the image so cold starts don't download them
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
