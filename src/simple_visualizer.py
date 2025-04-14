import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimpleVisualizer:
    """
    A simple 2D and 3D visualizer using Matplotlib to display body pose,
    hand landmarks, and face landmarks over time.
    """
    def __init__(self):
        """
        Initializes the visualizer by setting up necessary configurations.

        This includes:
        - Defining colors for different landmarks (pose, left hand, right hand, face).
        - Setting up connections for body pose and hand landmarks.
        
    
        """

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
        Visualizes the landmarks (pose, hands, and face) for each frame over time.
        The frames in the dictionary must contain:
        - "pose_landmarks": list of pose landmarks for each frame.
        - "left_hand_landmarks": list of left hand landmarks for each frame.
        - "right_hand_landmarks": list of right hand landmarks for each frame.
        - "face_landmarks": list of face landmarks for each frame.

        Args:
        landmarks_data (dict): A dictionary containing the frames of landmark data.
         """

        frames = landmarks_data["frames"]
        if not frames:
            print("No frames to visualize")
            return
            
        # Find the first frame with valid landmarks
        frame_idx = 0
        while frame_idx < len(frames) and not frames[frame_idx].get("pose_landmarks"):
            frame_idx += 1
            
        if frame_idx >= len(frames):
            print("No valid landmarks found in any frame")
            return
        
        # Set up the matplotlib figure with two subplots (front and side views)
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(121)  # Front view (X-Y)
        ax2 = fig.add_subplot(122, projection='3d')  # 3D view
        
        ax1.set_title("Front View (X-Y)")
        ax2.set_title("3D View")
        
        # Counter to track current frame
        current_frame = [0]
        
        def update(frame_idx):
            """
            Updates the visualization for each frame based on the current index.
            This function clears the previous plot and redraws the new frame.
            
            Args:
            frame_idx (int): The index of the frame to be visualized.
            
            Returns:
            tuple: The updated axes for 2D and 3D views (ax1, ax2).
            """
            ax1.clear()
            ax2.clear()
            
            # Set axis labels
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # Set axis limits
            ax1.set_xlim([0, 1])
            ax1.set_ylim([1, 0])  # Inverted Y-axis to match image coordinates
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.set_zlim([0, 1])
            
            # Set titles
            ax1.set_title(f"Front View - Frame {frame_idx}/{len(frames)-1}")
            ax2.set_title(f"3D View - Frame {frame_idx}/{len(frames)-1}")
            
            # Plot the landmarks
            self._plot_frame(ax1, ax2, frames[frame_idx])
            
            current_frame[0] = frame_idx
            
            return ax1, ax2
        
        # Create the slider
        from matplotlib.widgets import Slider
        ax_slider = plt.axes([0.15, 0.01, 0.7, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, 
                        valinit=0, valstep=1, valfmt='%d')
        
        def update_slider(val):
            """
            Handles the interaction with the slider to update the frame displayed
            based on the selected value.
            This function calls the update function to redraw the frame.
            
            Args:
            val (float): The current value of the slider, used to update the frame index.
            """
            frame_idx = int(slider.val)
            update(frame_idx)
            fig.canvas.draw_idle()
        
        slider.on_changed(update_slider)
        
        # Plot the initial frame
        update(0)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        plt.show()
    
    def _plot_frame(self, ax_2d, ax_3d, frame):
        """
        Plots the landmarks for a specific frame in both 2D and 3D views.
        This function calls different helper functions to plot the body pose,
        hand, and face landmarks for the given frame.
        
        Args:
        ax_2d (matplotlib.axes.Axes): The 2D plot axis.
        ax_3d (matplotlib.axes.Axes): The 3D plot axis.
        frame (dict): The current frame's landmark data.
        """
        if frame.get("pose_landmarks"):
            self._plot_pose(ax_2d, ax_3d, frame["pose_landmarks"])
        
        # Plot hand landmarks
        if frame.get("left_hand_landmarks"):
            self._plot_hand(ax_2d, ax_3d, frame["left_hand_landmarks"], is_left=True)
        
        if frame.get("right_hand_landmarks"):
            self._plot_hand(ax_2d, ax_3d, frame["right_hand_landmarks"], is_left=False)
        
        # Plot face landmarks
        if frame.get("face_landmarks"):
            self._plot_face(ax_2d, ax_3d, frame["face_landmarks"])
    
    def _plot_pose(self, ax_2d, ax_3d, pose_landmarks):
        
        """
        Plots the body pose landmarks and their connections in both 2D and 3D views.
        This function plots each landmark and connects them based on predefined pose connections.
        
        Args:
        ax_2d (matplotlib.axes.Axes): The 2D plot axis.
        ax_3d (matplotlib.axes.Axes): The 3D plot axis.
        pose_landmarks (list): The list of pose landmarks to be plotted.
        """
        points = np.array([[landmark["x"], landmark["y"], landmark["z"]]
                          for landmark in pose_landmarks])
        
        # Plot points in 2D (front view)
        ax_2d.scatter(points[:, 0], points[:, 1], color=self.colors["pose"], s=30)
        
        # Plot points in 3D
        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], color=self.colors["pose"], s=30)
        
        # Plot connections
        for connection in self.pose_connections:
            if connection[0] < len(pose_landmarks) and connection[1] < len(pose_landmarks):
                # 2D connections
                ax_2d.plot([points[connection[0], 0], points[connection[1], 0]],
                           [points[connection[0], 1], points[connection[1], 1]],
                           color=self.colors["pose"])
                
                # 3D connections
                ax_3d.plot([points[connection[0], 0], points[connection[1], 0]],
                           [points[connection[0], 1], points[connection[1], 1]],
                           [points[connection[0], 2], points[connection[1], 2]],
                           color=self.colors["pose"])
    
    def _plot_hand(self, ax_2d, ax_3d, hand_landmarks, is_left=True):
       
        color = self.colors["left_hand"] if is_left else self.colors["right_hand"]
        
        # Extract points
        points = np.array([[landmark["x"], landmark["y"], landmark["z"]]
                          for landmark in hand_landmarks])
        
        # Plot points in 2D (front view)
        ax_2d.scatter(points[:, 0], points[:, 1], color=color, s=20)
        
        # Plot points in 3D
        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=20)
        
        # Plot connections
        for connection in self.hand_connections:
            if connection[0] < len(hand_landmarks) and connection[1] < len(hand_landmarks):
                # 2D connections
                ax_2d.plot([points[connection[0], 0], points[connection[1], 0]],
                           [points[connection[0], 1], points[connection[1], 1]],
                           color=color)
                
                # 3D connections
                ax_3d.plot([points[connection[0], 0], points[connection[1], 0]],
                           [points[connection[0], 1], points[connection[1], 1]],
                           [points[connection[0], 2], points[connection[1], 2]],
                           color=color)
    
    def _plot_face(self, ax_2d, ax_3d, face_landmarks):
        """
    Plots the face landmarks in both 2D and 3D views.
    The function extracts the (x, y, z) coordinates from the provided face landmarks and 
    visualizes them in both 2D and 3D plots. It uses a predefined color for face landmarks (green).
    
    Args:
    ax_2d (matplotlib.axes.Axes): The 2D plot axis.
    ax_3d (matplotlib.axes.Axes): The 3D plot axis.
    face_landmarks (list): The list of face landmarks to be plotted. Each landmark contains 'x', 'y', and 'z' coordinates.
    """
        points = np.array([[landmark["x"], landmark["y"], landmark["z"]]
                          for landmark in face_landmarks])
        
        # Plot points in 2D (front view)
        ax_2d.scatter(points[:, 0], points[:, 1], color=self.colors["face"], s=10)
        
        # Plot points in 3D
        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], color=self.colors["face"], s=10)
