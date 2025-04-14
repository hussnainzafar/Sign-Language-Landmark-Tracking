import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm import tqdm
from .utils.depth_filter import DepthFilter
from .utils.orientation_calculator import OrientationCalculator

class LandmarkExtractor:
    """
    Extracts landmarks (pose, hands, face) from video frames using the MediaPipe Holistic model.

    Attributes:
        mp_holistic (mediapipe.solutions.holistic): MediaPipe Holistic model used for landmark extraction.
        mp_drawing (mediapipe.solutions.drawing_utils): Utilities for drawing landmarks on images.
        mp_drawing_styles (mediapipe.solutions.drawing_styles): Default styles for landmark drawing.
        depth_filter (DepthFilter): Custom depth filter used to smooth landmark coordinates.
        orientation_calculator (OrientationCalculator): Custom tool to calculate hand orientations.

    Methods:
        process_video(video_path: str, show_video: bool = False) -> dict:
            Processes the video frame by frame to extract landmarks and optionally displays the video.
        _extract_frame_landmarks(results: mediapipe.framework.formats.landmark_pb2.LandmarkList, frame_idx: int) -> dict:
            Extracts landmarks from a single frame and calculates hand orientations.
        _process_pose_landmarks(landmarks: mediapipe.framework.formats.landmark_pb2.LandmarkList) -> list:
            Processes pose landmarks and returns a list of x, y, z, and visibility values.
        _process_hand_landmarks(landmarks: mediapipe.framework.formats.landmark_pb2.LandmarkList) -> list:
            Processes hand landmarks and returns a list of x, y, z coordinates.
        _process_face_landmarks(landmarks: mediapipe.framework.formats.landmark_pb2.LandmarkList) -> list:
            Selects and processes specific important face landmarks.
        _draw_landmarks(image: numpy.ndarray, results: mediapipe.framework.formats.landmark_pb2.LandmarkList):
            Draws the landmarks on the given image for visual display.
    """
    
    def __init__(self):
        """
        Initializes the necessary components for landmark extraction, including MediaPipe Holistic model, 
        drawing utilities, and custom tools for depth filtering and orientation calculation.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.depth_filter = DepthFilter(window_size=5)
        self.orientation_calculator = OrientationCalculator()
    
    def process_video(self, video_path, show_video=False):
        """
        Processes a video frame by frame to extract landmarks using MediaPipe Holistic.
        Optionally displays the video with drawn landmarks.

        Args:
            video_path (str): Path to the video file to process.
            show_video (bool, optional): Whether to display the video with drawn landmarks. Defaults to False.

        Returns:
            dict: A dictionary containing metadata about the video and the extracted landmarks for each frame.
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
                
                frame_data = self._extract_frame_landmarks(results, frame_idx)
                landmarks_data["frames"].append(frame_data)
                
                if show_video:
                    self._draw_landmarks(frame, results)
                    cv2.imshow('MediaPipe Holistic', frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        landmarks_data = self.depth_filter.process_sequence(landmarks_data)
        return landmarks_data

    def _extract_frame_landmarks(self, results, frame_idx):
        """
        Extracts all types of landmarks (pose, hands, face) from a single frame.
        Also calculates hand orientation if hand landmarks are detected.

        Args:
            results (mediapipe.framework.formats.landmark_pb2.LandmarkList): The landmarks detected in the current frame.
            frame_idx (int): The index of the current frame in the video.

        Returns:
            dict: A dictionary containing landmarks for pose, hands, and face, as well as hand orientation data.
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

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.LandmarkList): The pose landmarks detected in the current frame.

        Returns:
            list: A list of dictionaries containing x, y, z, and visibility values for each pose landmark.
            None: If no pose landmarks are detected.
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

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.LandmarkList): The hand landmarks detected in the current frame.

        Returns:
            list: A list of dictionaries containing x, y, z coordinates for each hand landmark.
            None: If no hand landmarks are detected.
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

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.LandmarkList): The face landmarks detected in the current frame.

        Returns:
            list: A list of dictionaries containing x, y, z coordinates for important face landmarks.
            None: If no face landmarks are detected.
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

        Args:
            image (numpy.ndarray): The image frame on which landmarks are drawn.
            results (mediapipe.framework.formats.landmark_pb2.LandmarkList): The landmarks detected in the current frame.

        Returns:
            None
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
