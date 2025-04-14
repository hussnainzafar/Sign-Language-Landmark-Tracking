import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm import tqdm
from .utils.depth_filter import DepthFilter
from .utils.orientation_calculator import OrientationCalculator

class LandmarkExtractor:
    def __init__(self):
        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize depth filter for z-axis stability
        self.depth_filter = DepthFilter(window_size=5)
        
        # Initialize orientation calculator for quaternions
        self.orientation_calculator = OrientationCalculator()
        
    def process_video(self, video_path, show_video=False):
        """
        Process a video file and extract landmarks
        
        Args:
            video_path: Path to the video file
            show_video: Whether to show the video with landmarks overlay
            
        Returns:
            Dictionary containing landmark data for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize results dictionary
        landmarks_data = {
            "metadata": {
                "video_path": video_path,
                "frame_count": frame_count,
                "fps": fps,
                "duration": frame_count / fps if fps > 0 else 0,
                "timestamp": time.time()
            },
            "frames": []
        }
        
        # Process each frame
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2  # Use the most accurate model
        ) as holistic:
            
            for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = holistic.process(frame_rgb)
                
                # Extract landmarks
                frame_data = self._extract_frame_landmarks(results, frame_idx)
                landmarks_data["frames"].append(frame_data)
                
                # Show video with landmarks if requested
                if show_video:
                    self._draw_landmarks(frame, results)
                    cv2.imshow('MediaPipe Holistic', frame)
                    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                        break
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
            
        # Apply depth filtering across all frames
        landmarks_data = self.depth_filter.process_sequence(landmarks_data)
        
        return landmarks_data
    
    def _extract_frame_landmarks(self, results, frame_idx):
        """Extract landmarks from a single frame"""
        frame_data = {
            "frame_idx": frame_idx,
            "pose_landmarks": self._process_pose_landmarks(results.pose_landmarks),
            "left_hand_landmarks": self._process_hand_landmarks(results.left_hand_landmarks),
            "right_hand_landmarks": self._process_hand_landmarks(results.right_hand_landmarks),
            "face_landmarks": self._process_face_landmarks(results.face_landmarks)
        }
        
        # Calculate hand orientations if hand landmarks are available
        if results.left_hand_landmarks:
            frame_data["left_hand_orientation"] = self.orientation_calculator.calculate_hand_orientation(
                results.left_hand_landmarks, results.pose_landmarks, is_left_hand=True
            )
        
        if results.right_hand_landmarks:
            frame_data["right_hand_orientation"] = self.orientation_calculator.calculate_hand_orientation(
                results.right_hand_landmarks, results.pose_landmarks, is_left_hand=False
            )
            
        return frame_data
    
    def _process_pose_landmarks(self, landmarks):
        """Process pose landmarks"""
        if not landmarks:
            return None
            
        pose_data = []
        for idx, landmark in enumerate(landmarks.landmark):
            pose_data.append({
                "idx": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        return pose_data
    
    def _process_hand_landmarks(self, landmarks):
        """Process hand landmarks"""
        if not landmarks:
            return None
            
        hand_data = []
        for idx, landmark in enumerate(landmarks.landmark):
            hand_data.append({
                "idx": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
        return hand_data
    
    def _process_face_landmarks(self, landmarks):
        """Process face landmarks"""
        if not landmarks:
            return None
            
        # For efficiency, we'll only store a subset of face landmarks
        # that are most relevant for sign language (eyes, nose, mouth)
        important_indices = [
            0,   # Nose tip
            8,   # Chin
            61,  # Left eye inner corner
            291, # Right eye inner corner
            13,  # Left eye outer corner
            263, # Right eye outer corner
            78,  # Left mouth corner
            308  # Right mouth corner
        ]
        
        face_data = []
        for idx in important_indices:
            landmark = landmarks.landmark[idx]
            face_data.append({
                "idx": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
        return face_data
    
    def _draw_landmarks(self, image, results):
        """Draw landmarks on the image for visualization"""
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
        )