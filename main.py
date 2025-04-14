import os
import cv2
import numpy as np
import json
import argparse
import sys
from src.landmark_extractor import LandmarkExtractor
from src.landmark_visualizer import LandmarkVisualizer
from src.simple_visualizer import SimpleVisualizer
from src.utils.orientation_calculator import OrientationCalculator
import os 
from dotenv import load_dotenv

load_dotenv()

data_source=os.environ.get("DATA_PATH")

def process_single_video(input_path, output_path, show_video=False, visualize=False, simple_vis=False):
    print(f"Processing video: {input_path}")
    
    extractor = LandmarkExtractor()
    landmarks_data = extractor.process_video(input_path, show_video=show_video)
    
    with open(output_path, 'w') as f:
        json.dump(landmarks_data, f, indent=2)
    print(f"Landmarks saved to: {output_path}")
    
    if visualize:
        try:
            if simple_vis:
                print("Using simple 2D matplotlib visualizer...")
                visualizer = SimpleVisualizer()
                visualizer.visualize_landmarks(landmarks_data)
            else:
                print("Using 3D PyVista visualizer...")
                visualizer = LandmarkVisualizer()
                visualizer.visualize_landmarks(landmarks_data)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Trying simple matplotlib visualizer as fallback...")
            try:
                visualizer = SimpleVisualizer()
                visualizer.visualize_landmarks(landmarks_data)
            except Exception as e2:
                print(f"Simple visualizer also failed: {e2}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sign Language Landmark Tracker")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--no-show", action="store_true", help="Disable video preview")
    parser.add_argument("--simple", action="store_true", help="Use simple matplotlib visualizer instead of 3D")
    parser.add_argument("--file", type=str, help="Process a specific file instead of all files in the data folder")
    args = parser.parse_args()
    
    input_folder = data_source
    output_folder = "output"
    show_video = not args.no_show
    visualize = not args.no_vis
    simple_vis = args.simple

    os.makedirs(output_folder, exist_ok=True)
    
    # Process specific file if provided
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        video_name = os.path.splitext(os.path.basename(args.file))[0]
        output_path = os.path.join(output_folder, f"{video_name}.json")
        process_single_video(args.file, output_path, show_video, visualize, simple_vis)
    else:
        # Process all MP4 files in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".mp4"):
                input_path = os.path.join(input_folder, filename)
                video_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{video_name}.json")
                process_single_video(input_path, output_path, show_video, visualize, simple_vis)

if __name__ == "__main__":
    main()
