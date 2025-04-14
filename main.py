import os
import cv2
import numpy as np
import json
from src.landmark_extractor import LandmarkExtractor
from src.landmark_visualizer import LandmarkVisualizer
from src.utils.orientation_calculator import OrientationCalculator
import os 
from dotenv import load_dotenv

load_dotenv()

data_source=os.environ.get("DATA_PATH")

def process_single_video(input_path, output_path, show_video=False, visualize=False):
    print(f"Processing video: {input_path}")
    
    extractor = LandmarkExtractor()
    landmarks_data = extractor.process_video(input_path, show_video=show_video)
    
    with open(output_path, 'w') as f:
        json.dump(landmarks_data, f, indent=2)
    print(f"Landmarks saved to: {output_path}")
    
    if visualize:
        visualizer = LandmarkVisualizer()
        visualizer.visualize_landmarks(landmarks_data)

def main():
    input_folder = data_source
    output_folder = "output"
    show_video = True
    visualize = True

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{video_name}.json")
            process_single_video(input_path, output_path, show_video, visualize)

if __name__ == "__main__":
    main()
