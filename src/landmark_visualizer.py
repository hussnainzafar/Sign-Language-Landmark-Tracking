import numpy as np
import pyvista as pv
import time

class LandmarkVisualizer:
    """
    Visualizer for 3D landmarks. This class helps visualize pose, hand, and face landmarks 
    in 3D space using PyVista for interactive viewing. It supports real-time visualization 
    and navigation of landmarks data across frames.
    """
    
    def __init__(self):
        """
        Initializes the LandmarkVisualizer object with default settings. 
        This includes setting up predefined colors for different body parts 
        and specifying the connections for pose and hand landmarks.
        """
        self.colors = {
            "pose": "white",
            "left_hand": "red",
            "right_hand": "blue",
            "face": "green"
        }
        
        self.pose_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32)
        ]
        
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17)
        ]
    
    def visualize_landmarks(self, landmarks_data):
        """
        Visualizes the landmarks in a 3D plot using PyVista. 
        It takes landmark data and visualizes pose, hand, and face landmarks. 
        The function creates an interactive 3D viewer with a slider for frame navigation.

        Args:
            landmarks_data (dict): Contains frames with landmark data, including pose, hand, and face landmarks.

        Returns:
            None: Displays a 3D interactive visualization with a slider for frame navigation.
        """
        plotter = pv.Plotter()
        plotter.set_background("black")
        plotter.add_text("Sign Language Landmark Visualization", font_size=20)
        
        frames = landmarks_data["frames"]
        if not frames:
            print("No frames to visualize")
            return
        
        frame_idx = 0
        while frame_idx < len(frames) and not frames[frame_idx].get("pose_landmarks"):
            frame_idx += 1
        
        if frame_idx >= len(frames):
            print("No valid landmarks found in any frame")
            return
        
        self._setup_visualization(plotter, frames[frame_idx])
        
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
        
        plotter.show()
    
    def _setup_visualization(self, plotter, frame):
        """
        Sets up the 3D visualization, including camera position, axes, and the initial frame. 
        It prepares the plotter to display the 3D landmark data by configuring view settings. 
        This method ensures the scene is ready for interaction and visual updates.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            frame (dict): Contains landmark data for the current frame to be visualized.
        
        Returns:
            None: Configures the plotter and displays the frame.
        """
        self._visualize_frame(plotter, frame)
        
        plotter.camera_position = [
            (0, -2, 0),
            (0, 0, 0),
            (0, 0, 1)
        ]
        
        plotter.add_axes()
        plotter.enable_trackball_style()
    
    def _visualize_frame(self, plotter, frame):
        """
        Visualizes the landmarks for a single frame, including pose, hands, and face. 
        This method adds points and lines for each part of the body to the 3D plot. 
        It updates the plot with the landmarks of the current frame, allowing for real-time interaction.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            frame (dict): Landmark data for a single frame, including pose, hands, and face.
        
        Returns:
            None: Adds points and lines for pose, hands, and face to the plotter.
        """
        if frame.get("pose_landmarks"):
            self._visualize_pose(plotter, frame["pose_landmarks"])
        
        if frame.get("left_hand_landmarks"):
            self._visualize_hand(plotter, frame["left_hand_landmarks"], is_left=True)
        
        if frame.get("right_hand_landmarks"):
            self._visualize_hand(plotter, frame["right_hand_landmarks"], is_left=False)
        
        if frame.get("face_landmarks"):
            self._visualize_face(plotter, frame["face_landmarks"])
        
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
        """
        Visualizes the pose landmarks with connections for body joints. 
        This method draws lines between key landmarks to represent the body structure in 3D space. 
        It allows for a clear view of body movement and positioning in each frame.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            pose_landmarks (list): List of pose landmarks with 'x', 'y', 'z' coordinates.
        
        Returns:
            None: Adds points and lines for the body pose to the plotter.
        """
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in pose_landmarks
        ])
        
        points = self._scale_and_center_points(points)
        
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=self.colors["pose"], point_size=10)
        
        for connection in self.pose_connections:
            if connection[0] < len(pose_landmarks) and connection[1] < len(pose_landmarks):
                line = pv.Line(points[connection[0]], points[connection[1]])
                plotter.add_mesh(line, color=self.colors["pose"], line_width=3)
    
    def _visualize_hand(self, plotter, hand_landmarks, is_left=True):
        """
        Visualizes the hand landmarks in 3D with connections between fingers and palm. 
        This method draws lines to connect the landmarks, showing the structure of the hand. 
        It supports both left and right hands and renders them in different colors.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            hand_landmarks (list): List of hand landmarks with 'x', 'y', 'z' coordinates.
            is_left (bool): True if visualizing the left hand, False if right.
        
        Returns:
            None: Adds points and lines for hand landmarks to the plotter.
        """
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in hand_landmarks
        ])
        
        points = self._scale_and_center_points(points)
        
        color = self.colors["left_hand"] if is_left else self.colors["right_hand"]
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=color, point_size=8)
        
        for connection in self.hand_connections:
            if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                line = pv.Line(points[connection[0]], points[connection[1]])
                plotter.add_mesh(line, color=color, line_width=2)
    
    def _visualize_face(self, plotter, face_landmarks):
        """
        Visualizes the face landmarks in 3D as a set of points with no connections. 
        This method plots each facial landmark as a point in 3D space. 
        It provides a visualization of the facial structure.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            face_landmarks (list): List of face landmarks with 'x', 'y', 'z' coordinates.
        
        Returns:
            None: Adds points for face landmarks to the plotter.
        """
        points = np.array([
            [landmark["x"], landmark["y"], landmark["z"]]
            for landmark in face_landmarks
        ])
        
        points = self._scale_and_center_points(points)
        
        point_cloud = pv.PolyData(points)
        plotter.add_points(point_cloud, color=self.colors["face"], point_size=5)
    
    def _visualize_hand_orientation(self, plotter, hand_landmarks, orientation, is_left=True):
        """
        Visualizes the orientation of the hand, showing the rotation axes and Euler angles. 
        This method draws rotation axes from the wrist and displays the Euler angles 
        (pitch, yaw, roll) that describe the hand's orientation in 3D space.

        Args:
            plotter (pyvista.Plotter): The plotter object to display the scene.
            hand_landmarks (list): List of hand landmarks with 'x', 'y', 'z' coordinates.
            orientation (dict): Hand orientation data including rotation matrix and Euler angles.
            is_left (bool): True if visualizing the left hand, False if right.
        
        Returns:
            None: Adds rotation axes and Euler angle text to the plotter.
        """
        wrist_pos = np.array([hand_landmarks[0]["x"], hand_landmarks[0]["y"], hand_landmarks[0]["z"]])
        wrist_pos = self._scale_and_center_points(np.array([wrist_pos]))[0]
        
        rotation_matrix = np.array(orientation["rotation_matrix"])
        
        scale = 0.1
        x_axis = wrist_pos + rotation_matrix[:, 0] * scale
        y_axis = wrist_pos + rotation_matrix[:, 1] * scale
        z_axis = wrist_pos + rotation_matrix[:, 2] * scale
        
        color = self.colors["left_hand"] if is_left else self.colors["right_hand"]
        
        x_line = pv.Line(wrist_pos, x_axis)
        y_line = pv.Line(wrist_pos, y_axis)
        z_line = pv.Line(wrist_pos, z_axis)
        
        plotter.add_mesh(x_line, color="red", line_width=3)
        plotter.add_mesh(y_line, color="green", line_width=3)
        plotter.add_mesh(z_line, color="blue", line_width=3)
        
        text = f"Euler Angles: Pitch = {orientation['pitch']:.2f}, Yaw = {orientation['yaw']:.2f}, Roll = {orientation['roll']:.2f}"
        plotter.add_text(text, position="lower_left", color="white", font_size=12)
    
    def _scale_and_center_points(self, points):
        """
        Scales and centers the points to fit into the visualization window. 
        This method ensures that the landmark data is adjusted properly to display 
        within the bounds of the 3D plot.

        Args:
            points (np.array): Array of points to be scaled and centered.
        
        Returns:
            np.array: Scaled and centered points.
        """
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale
        return points
