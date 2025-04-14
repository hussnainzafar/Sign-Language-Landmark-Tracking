import numpy as np
import pyvista as pv
import time

class LandmarkVisualizer:
    """
    Visualizer for 3D landmarks
    """
    def __init__(self):
        # Define colors
        self.colors = {
            "pose": "white",
            "left_hand": "red",
            "right_hand": "blue",
            "face": "green"
        }
        
        # Define connections for pose
        self.pose_connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15),
            # Right arm
            (12, 14), (14, 16),
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32)
        ]
        
        # Define connections for hands
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # Palm
        ]
    
    def visualize_landmarks(self, landmarks_data):
        """
    Visualize the landmarks in a 3D PyVista plot.
    Initializes the scene and slider for navigating through frames.
    Displays the first valid frame and updates as user slides.
    """
        # Create a PyVista plotter
        plotter = pv.Plotter()
        plotter.set_background("black")
        
        # Add a title
        plotter.add_text("Sign Language Landmark Visualization", font_size=20)
        
        # Get the first frame with valid landmarks
        frames = landmarks_data["frames"]
        if not frames:
            print("No frames to visualize")
            return
        
        # Find the first frame with valid pose landmarks
        frame_idx = 0
        while frame_idx < len(frames) and not frames[frame_idx].get("pose_landmarks"):
            frame_idx += 1
        
        if frame_idx >= len(frames):
            print("No valid landmarks found in any frame")
            return
        
        # Set up the visualization
        self._setup_visualization(plotter, frames[frame_idx])
        
        # Add a slider for frame selection
        def update_frame(value):
            frame_idx = int(value)
            if 0 <= frame_idx < len(frames):
                plotter.clear_actors()
                self._visualize_frame(plotter, frames[frame_idx])
                plotter.update()
        
        plotter.add_slider_widget(
            update_frame,
            [0, len(frames) - 1],
            title="Frame",
            value=0,
            pointa=(0.1, 0.1),
            pointb=(0.9, 0.1),
            style="modern"
        )
        
        # Show the plotter
        plotter.show()
    
    def _setup_visualization(self, plotter, frame):
        
    #Set up the 3D view with camera, axes, and initial frame.
    #Uses the first valid frame to visualize landmarks.
    #Enables trackball interaction for better view control.

        self._visualize_frame(plotter, frame)
        
        # Set up camera
        plotter.camera_position = [
            (0, -2, 0),  # Camera position
            (0, 0, 0),   # Focal point
            (0, 0, 1)    # Up direction
        ]
        
        # Add axes
        plotter.add_axes()
        
        # Enable trackball camera mode
        plotter.enable_trackball_style()
    
    def _visualize_frame(self, plotter, frame):
    
    # Visualizes a single frame's landmarks (pose, hands, face).
    # Calls helper functions to draw each body part if available.
    # Also adds hand orientation axes if data is present.
    
        # Visualize pose landmarks
        if frame.get("pose_landmarks"):
            self._visualize_pose(plotter, frame["pose_landmarks"])
        
        # Visualize hand landmarks
        if frame.get("left_hand_landmarks"):
            self._visualize_hand(plotter, frame["left_hand_landmarks"], is_left=True)
        
        if frame.get("right_hand_landmarks"):
            self._visualize_hand(plotter, frame["right_hand_landmarks"], is_left=False)
        
        # Visualize face landmarks
        if frame.get("face_landmarks"):
            self._visualize_face(plotter, frame["face_landmarks"])
        
        # Visualize hand orientations
        if frame.get("left_hand_orientation") and frame.get("left_hand_landmarks"):
            self._visualize_hand_orientation(
                plotter, 
                frame["left_hand_landmarks"], 
                frame["left_hand_orientation"],
                is_left=True
            )
        
        if frame.get("right_hand_orientation") and frame.get("right_hand_landmarks"):
            self._visualize_hand_orientation(
                plotter, 
                frame["right_hand_landmarks"], 
                frame["right_hand_orientation"],
                is_left=False
            )
    
    def _visualize_pose(self, plotter, pose_landmarks):
    #      Visualizes the body pose landmarks in 3D.
    # Adds point cloud and lines for body joint connections.
    # Scales and centers the pose data for consistent display.
        # Extract points
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in pose_landmarks
        ])
        
        # Scale and center the points
        points = self._scale_and_center_points(points)
        
        # Add points
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=self.colors["pose"], point_size=10)
        
        # Add lines for connections
        for connection in self.pose_connections:
            if connection[0] < len(pose_landmarks) and connection[1] < len(pose_landmarks):
                line = pv.Line(points[connection[0]], points[connection[1]])
                plotter.add_mesh(line, color=self.colors["pose"], line_width=3)
    
    def _visualize_hand(self, plotter, hand_landmarks, is_left=True):
    #      Draws the hand landmarks with finger and palm connections.
    # Selects color based on left or right hand.
    # Centers and scales the points for 3D visualization.
        # Extract points
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in hand_landmarks
        ])
        
        # Scale and center the points
        points = self._scale_and_center_points(points)
        
        # Add points
        color = self.colors["left_hand"] if is_left else self.colors["right_hand"]
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=color, point_size=8)
        
        # Add lines for connections
        for connection in self.hand_connections:
            if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                line = pv.Line(points[connection[0]], points[connection[1]])
                plotter.add_mesh(line, color=color, line_width=2)
    
    def _visualize_face(self, plotter, face_landmarks):
    #      Plots 3D face landmarks as a point cloud.
    # No connections—just individual landmark dots.
    # Scales and centers points for correct positioning.
        # Extract points
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in face_landmarks
        ])
        
        # Scale and center the points
        points = self._scale_and_center_points(points)
        
        # Add points
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=self.colors["face"], point_size=5)
    
    def _visualize_hand_orientation(self, plotter, hand_landmarks, orientation, is_left=True):
    #      Normalizes points by centering them around origin.
    # Scales points based on max distance to keep visuals consistent.
    # Ensures that different body parts fit well in the same 3D scene.
        # Get wrist position
        wrist_pos = np.array([
            hand_landmarks[0]["x"],
            hand_landmarks[0]["y"],
            hand_landmarks[0]["z"]
        ])
        
        # Scale and center the wrist position
        wrist_pos = self._scale_and_center_points(np.array([wrist_pos]))[0]
        
        # Get rotation matrix
        rotation_matrix = np.array(orientation["rotation_matrix"])
        
        # Scale for visualization
        scale = 0.1
        
        # Create axes
        x_axis = wrist_pos + rotation_matrix[:, 0] * scale
        y_axis = wrist_pos + rotation_matrix[:, 1] * scale
        z_axis = wrist_pos + rotation_matrix[:, 2] * scale
        
        # Add lines for axes
        color = self.colors["left_hand"] if is_left else self.colors["right_hand"]
        
        # X-axis (red)
        x_line = pv.Line(wrist_pos, x_axis)
        plotter.add_mesh(x_line, color="red", line_width=3)
        
        # Y-axis (green)
        y_line = pv.Line(wrist_pos, y_axis)
        plotter.add_mesh(y_line, color="green", line_width=3)
        
        # Z-axis (blue)
        z_line = pv.Line(wrist_pos, z_axis)
        plotter.add_mesh(z_line, color="blue", line_width=3)
        
        # Add text for Euler angles
        euler_angles = orientation["euler_angles"]
        text = f"Pitch: {euler_angles['pitch']:.1f}°\nYaw: {euler_angles['yaw']:.1f}°\nRoll: {euler_angles['roll']:.1f}°"
        
        # Position the text near the wrist
        text_pos = wrist_pos + np.array([0, 0, 0.1])
        plotter.add_point_labels([text_pos], [text], font_size=10, shape=None)
    
    def _scale_and_center_points(self, points):
        """Scale and center points for better visualization"""
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Scale the points
        max_distance = np.max(np.linalg.norm(centered_points, axis=1))
        if max_distance > 0:
            scaled_points = centered_points / max_distance
        else:
            scaled_points = centered_points
        
        return scaled_points