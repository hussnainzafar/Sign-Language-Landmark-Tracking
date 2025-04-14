import numpy as np
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat

class OrientationCalculator:
    """
    Calculator for hand orientation using quaternions
    """
    def __init__(self):
        # Define hand landmark indices
        self.WRIST = 0
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.PINKY_MCP = 17
        
        # Define pose landmark indices
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
    
    def calculate_hand_orientation(self, hand_landmarks, pose_landmarks, is_left_hand=True):
        """
        Calculate hand orientation using quaternions
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            pose_landmarks: MediaPipe pose landmarks
            is_left_hand: Whether the hand is the left hand
            
        Returns:
            Dictionary containing quaternion, euler angles, and rotation matrix
        """
        if not hand_landmarks or not pose_landmarks:
            return None
        
        # Get wrist position
        wrist = np.array([
            hand_landmarks.landmark[self.WRIST].x,
            hand_landmarks.landmark[self.WRIST].y,
            hand_landmarks.landmark[self.WRIST].z
        ])
        
        # Get index, middle, and pinky MCP positions to define the hand plane
        index_mcp = np.array([
            hand_landmarks.landmark[self.INDEX_MCP].x,
            hand_landmarks.landmark[self.INDEX_MCP].y,
            hand_landmarks.landmark[self.INDEX_MCP].z
        ])
        
        middle_mcp = np.array([
            hand_landmarks.landmark[self.MIDDLE_MCP].x,
            hand_landmarks.landmark[self.MIDDLE_MCP].y,
            hand_landmarks.landmark[self.MIDDLE_MCP].z
        ])
        
        pinky_mcp = np.array([
            hand_landmarks.landmark[self.PINKY_MCP].x,
            hand_landmarks.landmark[self.PINKY_MCP].y,
            hand_landmarks.landmark[self.PINKY_MCP].z
        ])
        
        # Get elbow position from pose landmarks
        elbow_idx = self.LEFT_ELBOW if is_left_hand else self.RIGHT_ELBOW
        elbow = np.array([
            pose_landmarks.landmark[elbow_idx].x,
            pose_landmarks.landmark[elbow_idx].y,
            pose_landmarks.landmark[elbow_idx].z
        ])
        
        # Calculate hand orientation
        # Z-axis: from wrist to middle MCP (forward direction)
        z_axis = middle_mcp - wrist
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X-axis: perpendicular to the plane formed by index, middle, and pinky MCPs
        # (pointing to the side of the hand)
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        x_axis = np.cross(v1, v2)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: perpendicular to both X and Z axes (pointing up from the palm)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Ensure X-axis is perpendicular to Y and Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert rotation matrix to quaternion
        quaternion = mat2quat(rotation_matrix)
        
        # Convert rotation matrix to Euler angles (in radians)
        euler_angles = mat2euler(rotation_matrix)
        
        # Convert Euler angles to degrees
        euler_degrees = np.degrees(euler_angles)
        
        return {
            "quaternion": quaternion.tolist(),
            "euler_angles": {
                "pitch": float(euler_degrees[0]),  # X-axis rotation
                "yaw": float(euler_degrees[1]),    # Y-axis rotation
                "roll": float(euler_degrees[2])    # Z-axis rotation
            },
            "rotation_matrix": rotation_matrix.tolist()
        }