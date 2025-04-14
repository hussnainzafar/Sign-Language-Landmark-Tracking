import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm import tqdm
from .utils.depth_filter import DepthFilter
from .utils.orientation_calculator import OrientationCalculator

class LandmarkExtractor:
    def __init__(self):
        # Initializes MediaPipe Holistic model, drawing utilities, and custom tools
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.depth_filter = DepthFilter(window_size=5)
        self.orientation_calculator = OrientationCalculator()
    
    def process_video(self, video_path, show_video=False):
        """
        Processes a video frame by frame to extract landmarks using MediaPipe Holistic.
        Optionally displays the video with drawn landmarks.
        Returns all the extracted data along with video metadata.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
        
        # Initializes the MediaPipe Holistic model and processes each video frame
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        ) as holistic:
            for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                # Extracts landmark data from the current frame
                frame_data = self._extract_frame_landmarks(results, frame_idx)
                landmarks_data["frames"].append(frame_data)
                
                # Optionally displays the video with drawn landmarks
                if show_video:
                    self._draw_landmarks(frame, results)
                    cv2.imshow('MediaPipe Holistic', frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Applies depth filtering to smoothen landmark coordinates
        landmarks_data = self.depth_filter.process_sequence(landmarks_data)
        return landmarks_data

    def _extract_frame_landmarks(self, results, frame_idx):
        """
        Extracts all types of landmarks (pose, hands, face) from a single frame.
        Also calculates hand orientation if hand landmarks are detected.
        """
        frame_data = {
            "frame_idx": frame_idx,
            "pose_landmarks": self._process_pose_landmarks(results.pose_landmarks),
            "left_hand_landmarks": self._process_hand_landmarks(results.left_hand_landmarks),
            "right_hand_landmarks": self._process_hand_landmarks(results.right_hand_landmarks),
            "face_landmarks": self._process_face_landmarks(results.face_landmarks)
        }
        
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
        """
        Processes pose landmarks and returns a list of x, y, z, and visibility values.
        Returns None if no landmarks are detected.
        """
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
        """
        Processes hand landmarks and returns a list of x, y, z coordinates.
        Returns None if no landmarks are detected.
        """
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
        """
        Selects and processes specific important face landmarks.
        Returns a list of selected landmark coordinates or None if not detected.
        """
        if not landmarks:
            return None
            
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
        """
        Draws pose and hand landmarks on the image using MediaPipe drawing tools.
        Used for visual display when show_video is enabled.
        """
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
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
