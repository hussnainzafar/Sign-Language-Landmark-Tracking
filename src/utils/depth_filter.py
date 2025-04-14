import numpy as np
from scipy.signal import savgol_filter

class DepthFilter:
    """
    Filter for improving depth (z-axis) accuracy and reducing jitter
    """
    def __init__(self, window_size=5, polyorder=2):
        self.window_size = window_size
        self.polyorder = polyorder
    
    def process_sequence(self, landmarks_data):
        """
        Process a sequence of landmarks to improve depth accuracy
        
        Args:
            landmarks_data: Dictionary containing landmark data for each frame
            
        Returns:
            Processed landmark data with improved depth values
        """
        frames = landmarks_data["frames"]
        if not frames:
            return landmarks_data
        
        # Process pose landmarks
        self._filter_landmark_sequence(frames, "pose_landmarks")
        
        # Process hand landmarks
        self._filter_landmark_sequence(frames, "left_hand_landmarks")
        self._filter_landmark_sequence(frames, "right_hand_landmarks")
        
        return landmarks_data
    
    def _filter_landmark_sequence(self, frames, landmark_type):
        """
        Filter a specific type of landmarks across all frames
        
        Args:
            frames: List of frame data
            landmark_type: Type of landmark to filter (e.g., "pose_landmarks")
        """
        # Check if we have enough frames for filtering
        if len(frames) < self.window_size:
            return
        
        # Check if the landmark type exists in the first frame
        if frames[0].get(landmark_type) is None:
            return
        
        # Get the number of landmarks in the first frame
        first_valid_frame_idx = 0
        while first_valid_frame_idx < len(frames) and frames[first_valid_frame_idx].get(landmark_type) is None:
            first_valid_frame_idx += 1
        
        if first_valid_frame_idx >= len(frames):
            return
        
        num_landmarks = len(frames[first_valid_frame_idx][landmark_type])
        
        # For each landmark index
        for landmark_idx in range(num_landmarks):
            # Extract z values across all frames
            z_values = []
            frame_indices = []
            
            for frame_idx, frame in enumerate(frames):
                if frame.get(landmark_type) is not None and landmark_idx < len(frame[landmark_type]):
                    z_values.append(frame[landmark_type][landmark_idx]["z"])
                    frame_indices.append(frame_idx)
            
            if len(z_values) < self.window_size:
                continue
            
            # Apply Savitzky-Golay filter to smooth z values
            z_values_np = np.array(z_values)
            
            # Adjust window size if needed
            window_size = min(self.window_size, len(z_values))
            if window_size % 2 == 0:
                window_size -= 1
            
            polyorder = min(self.polyorder, window_size - 1)
            
            if window_size > polyorder:
                z_filtered = savgol_filter(z_values_np, window_size, polyorder)
                
                # Update z values in the original data
                for i, frame_idx in enumerate(frame_indices):
                    if frames[frame_idx].get(landmark_type) is not None and landmark_idx < len(frames[frame_idx][landmark_type]):
                        frames[frame_idx][landmark_type][landmark_idx]["z"] = float(z_filtered[i])
    
    def kalman_filter_1d(self, measurements, process_variance=1e-5, measurement_variance=1e-1):
        """
        Apply a 1D Kalman filter to a sequence of measurements
        
        Args:
            measurements: List of measurements
            process_variance: Process variance
            measurement_variance: Measurement variance
            
        Returns:
            Filtered measurements
        """
        n = len(measurements)
        if n == 0:
            return []
        
        # Initial state
        x_hat = measurements[0]
        p = 1.0
        
        # Storage for filtered values
        filtered_values = [x_hat]
        
        # Process the remaining measurements
        for z in measurements[1:]:
            # Prediction update
            p = p + process_variance
            
            # Measurement update
            k = p / (p + measurement_variance)
            x_hat = x_hat + k * (z - x_hat)
            p = (1 - k) * p
            
            filtered_values.append(x_hat)
        
        return filtered_values