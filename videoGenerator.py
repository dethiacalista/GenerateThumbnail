from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import requests
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from urllib.parse import urlparse

# Initialize the FastAPI app
app = FastAPI()

# Mount the 'thumbnails' folder to serve files at '/thumbnails'
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Directory to store downloaded videos and generated thumbnails
VIDEO_DIR = "videos"
THUMBNAIL_DIR = "thumbnails"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# Pydantic model for input validation
class VideoRequest(BaseModel):
    video_url: HttpUrl  # Ensure the input is a valid URL

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

def download_video(video_url: str) -> str:
    """Download video from URL and save it locally."""
    try:
        # Extract file name from URL
        parsed_url = urlparse(video_url)
        video_name = os.path.basename(parsed_url.path)
        if not video_name:
            raise ValueError("Invalid video URL. Unable to extract file name.")

        video_extension = Path(video_name).suffix.lower()
        if video_extension not in [".mp4", ".avi", ".mov", ".mkv"]:
            raise HTTPException(status_code=400, detail="Unsupported video format.")

        video_path = os.path.join(VIDEO_DIR, video_name)
        if os.path.exists(video_path):
            logging.info(f"Video already exists locally: {video_path}")
            return video_path

        # Download video from URL
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Video downloaded successfully: {video_path}")
        return video_path
    except Exception as e:
        logging.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading video: {str(e)}")

def extract_key_frames(video_path: str, num_frames: int = 10) -> list:
    """Extract key frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise HTTPException(status_code=400, detail="Video has no frames.")

    frames = []
    step = max(1, frame_count // num_frames)

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if len(frames) >= num_frames:
            break

    cap.release()
    if not frames:
        raise HTTPException(status_code=400, detail="No valid frames extracted from the video.")
    
    logging.info(f"Extracted {len(frames)} key frames from video: {video_path}")
    return frames

def create_animated_gif(frames: list, output_path: str):
    """Create an animated GIF from a list of frames."""
    try:
        clip = ImageSequenceClip(frames, fps=2)
        clip.write_gif(output_path, fps=2)
        logging.info(f"Animated GIF created successfully: {output_path}")
    except Exception as e:
        logging.error(f"Error creating animated GIF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating animated GIF: {str(e)}")

@app.post("/generate-thumbnail/", response_class=FileResponse)
def generate_thumbnail_endpoint(request: VideoRequest):
    """API endpoint to generate a dynamic video thumbnail."""
    video_url = str(request.video_url)  # Convert HttpUrl to string
    parsed_url = urlparse(video_url)
    video_name = os.path.basename(parsed_url.path)
    thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{Path(video_name).stem}.gif")

    if os.path.exists(thumbnail_path):
        logging.info(f"Thumbnail already exists, returning cached file: {thumbnail_path}")
        return FileResponse(thumbnail_path)

    try:
        video_path = download_video(video_url)
        frames = extract_key_frames(video_path)
        create_animated_gif(frames, thumbnail_path)
        return FileResponse(thumbnail_path)
    except Exception as e:
        logging.error(f"Error generating thumbnail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating thumbnail: {str(e)}")

@app.get("/")
def root():
    return {"message": "Dynamic Video Thumbnail Generator is running."}
