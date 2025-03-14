###########################################################################################################
# -----------------------------------------     Version 1.3     ----------------------------------------- #
# Functions:                                                                                              #
# create_db(): Creates an SQLite database for storing user credentials. No parameters. No return value.   #
# LoginWindow: Manages the login window.                                                                  #
#     __init__(root: tk.Tk, video_path: str): Initializes the login window.                               #
#     create_widgets(): Creates login UI elements.                                                        #
#     login(): Validates user credentials. No parameters. No return value.                                #
#     open_create_account(): Opens account creation window. No parameters. No return value.               #
#     open_video_window(username: str): Opens lane detection app for a logged-in user.                    #
# CreateAccountWindow: Handles account creation.                                                          #
#     __init__(root: tk.Tk, login_window: LoginWindow): Initializes account creation window.              #
#     create_widgets(): Creates account creation UI elements.                                             #
#     create_account(): Validates and stores new user credentials.                                        #
# LaneDetectionApp: Processes video feed for lane detection.                                              #
#     __init__(root: tk.Tk, video_path: str, username: str): Initializes the lane detection GUI.          #
#     process_frame(self, frame): Processes the input video frame for display                             #
#     lane_detection(self, frame): Performs lane detection on the input frame                             #
#         average_lane(lines, prev_fit): Averages lane lines to maintain smooth lane tracking.            #
#         draw_lane_line(img, fit, color): Draws a lane line on the image overaly feed.                   #
#             rotate_image(image: ndarray, angle: float): Rotates the compass rose by the proper angle.   #
#     update_frame(): Updates the frames to the Tkinter GUI. No parameters. No return value.              #
#     start_video(): Starts the video processing and frame update loop. No parameters. No return value.   #
#     stop_video(): Stops the video processing and frame update loop. No parameters. No return value.     #
#     button_action(direction: str): Handles button presses for play, stop, and updates them in the log.  #
###########################################################################################################

import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Entry, messagebox
from PIL import Image, ImageTk
import sqlite3
import time
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# This declares global variables for lane smoothing
left_fit_avg = None
right_fit_avg = None
alpha = 0.1  # This sets the smoothing factor

# SQLite database setup
def create_db():
    """
    Creates an SQLite database for storing user credentials.

    This function connects to the 'user_credentials.db' database,
    creates a 'users' table if it doesn't already exist, and closes the connection.
    """
    conn = sqlite3.connect('user_credentials.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Window to login
class LoginWindow:
    """
    A class to create and manage the login window.

    Attributes:
        root: The main Tkinter window.
        video_path: Path to the video file used in the LaneDetectionApp.
    """
    def __init__(self, root, video_path):
        """
        Initializes the LoginWindow class with the given Tkinter window and video path.

        Args:
            root (tk.Tk): The main Tkinter window.
            video_path (str): Path to the video file.
        """
        self.root = root
        self.video_path = video_path
        self.root.title("Login")
        self.create_widgets()

    def create_widgets(self):
        """
        Creates and packs the widgets (labels, entry fields, and buttons) for the login window.
        """
        self.username_label = Label(self.root, text="Username:")
        self.username_label.pack()

        self.username_entry = Entry(self.root)
        self.username_entry.pack()

        self.password_label = Label(self.root, text="Password:")
        self.password_label.pack()

        self.password_entry = Entry(self.root, show="*")
        self.password_entry.pack()

        self.login_button = Button(self.root, text="Login", command=self.login)
        self.login_button.pack()

        self.create_account_button = Button(self.root, text="Create Account", command=self.open_create_account)
        self.create_account_button.pack()

    def login(self):
        """
        Handles the login process by checking the entered username and password.

        If the credentials match, the login window is closed and the video window is opened.
        If invalid, an error message is shown.
        """
        username = self.username_entry.get()
        password = self.password_entry.get()

        conn = sqlite3.connect('user_credentials.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            self.root.destroy()  # This closes the login window
            self.open_video_window(username)  # This opens the video window and passes the username
        else:
            messagebox.showerror("Error", "Invalid username or password")

    def open_create_account(self):
        """
        Opens the Create Account window when the "Create Account" button is pressed.

        The login window is hidden, and a new CreateAccountWindow is created.
        """
        self.root.withdraw()  # This hides the login window
        create_account_window = tk.Toplevel(self.root)  # This creates a new window
        CreateAccountWindow(create_account_window, self.root)  # This opens the CreateAccountWindow

    def open_video_window(self, username):
        """
        Opens the video window where the LaneDetectionApp is shown.

        Args:
            username (str): The username passed from the login window.
        """
        root = tk.Tk()
        app = LaneDetectionApp(root, self.video_path, username)
        root.mainloop()

# Create Account Window
class CreateAccountWindow:
    """
    A class to create and manage the create account window.

    Attributes:
        root: The main Tkinter window.
        login_window: Reference to the login window.
    """
    def __init__(self, root, login_window):
        """
        Initializes the CreateAccountWindow class with the given Tkinter window and login window reference.

        Args:
            root (tk.Tk): The main Tkinter window for creating an account.
            login_window (LoginWindow): Reference to the login window.
        """
        self.root = root
        self.login_window = login_window  # This stores the reference to the login window
        self.root.title("Create Account")
        self.create_widgets()

    def create_widgets(self):
        """
        Creates and packs the widgets (labels, entry fields, and buttons) for the create account window.
        """
        self.username_label = Label(self.root, text="Username:")
        self.username_label.pack()

        self.username_entry = Entry(self.root)
        self.username_entry.pack()

        self.password_label = Label(self.root, text="Password:")
        self.password_label.pack()

        self.password_entry = Entry(self.root, show="*")
        self.password_entry.pack()

        self.confirm_password_label = Label(self.root, text="Confirm Password:")
        self.confirm_password_label.pack()

        self.confirm_password_entry = Entry(self.root, show="*")
        self.confirm_password_entry.pack()

        self.create_account_button = Button(self.root, text="Create Account", command=self.create_account)
        self.create_account_button.pack()

    def create_account(self):
        """
        Handles the account creation process by validating the entered username and password.

        If the passwords match and the username is unique, the new account is created.
        Otherwise, an error message is shown.
        """
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        if not username or not password:
            messagebox.showerror("Error", "Username and password cannot be empty")
            return

        conn = sqlite3.connect("user_credentials.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            messagebox.showerror("Error", "Username already exists")
            return

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        messagebox.showinfo("Success", "Account created successfully")

        self.root.destroy()  # This closes the create account window
        if self.login_window:  # This shows the login window again if it exists
            self.login_window.deiconify()  # This makes the login window visible again

# Lane Detection App
class LaneDetectionApp:
    """
    LaneDetectionApp class handles the GUI for lane detection, video processing, and user interaction.

    Attributes:
        root: Tkinter window that holds the application GUI.
        cap: Video capture object for reading video files.
        display_width: Width of the video display window.
        display_height: Height of the video display window.
        fps: Frames per second from the video source.
        wait_time: Time to wait between frames based on FPS.
        top_frame: Tkinter Label for the top video display.
        bottom_frame: Tkinter Label for the bottom video display.
        control_frame: Frame that holds control buttons.
        log_frame: Frame that holds the action log display.
        running: Boolean flag indicating whether the app is currently processing video.
    """

    def __init__(self, root, video_path, username):
        """
        Initializes the LaneDetectionApp.

        Args:
            root: Tkinter window object for GUI.
            video_path: Path to the video file.
            username: Username of the logged-in user.
        """
        self.root = root
        self.root.title("Lane Detection GUI")
        self.username = username  # This stores the logged-in username

        self.cap = cv2.VideoCapture(video_path)
        self.display_width = 1083
        self.display_height = 500
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        # This gets the FPS from the video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.wait_time = int(1000 / (self.fps * 1.5)) if self.fps > 0 else 30

        # This creates frames for video feeds
        self.top_frame = Label(self.root)
        self.bottom_frame = Label(self.root)

        # Here, this grids the layout and places the video windows on the left side
        self.top_frame.grid(row=0, column=0, padx=10, pady=10)
        self.bottom_frame.grid(row=1, column=0, padx=10, pady=10)

        # This creates placeholders (black screen) for video windows
        black_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        black_image = Image.fromarray(black_frame)
        black_image_tk = ImageTk.PhotoImage(image=black_image)

        self.top_frame.imgtk = black_image_tk
        self.top_frame.configure(image=black_image_tk)

        self.bottom_frame.imgtk = black_image_tk
        self.bottom_frame.configure(image=black_image_tk)

        # This creates the control panel and buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        # Here, this sets row and column weights for layout, dividing the window into two halves
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=2)

        # This positions buttons in the top half of the right side (column 1, rows 0 and 1)
        button_styles = {
            "↑": ("FORWARD", 0, 0, "white", "#1976D2"),
            "←": ("LEFT", 1, 0, "white", "#8E24AA"),
            "STOP": ("STOP", 1, 1, "white", "#D32F2F"),
            "▶": ("PLAY", 0, 1, "black", "#FBC02D"),
            "→": ("RIGHT", 0, 2, "white", "#388E3C"),
            "↓": ("BACKWARD", 1, 2, "white", "#F57C00"),
        }

        # Here, this creates buttons and positions them in the top half of the right side
        for text, (direction, row, col, fg, bg) in button_styles.items():
            button = Button(self.control_frame, text=text, anchor="ne", fg=fg, bg=bg, command=lambda d=direction: self.button_action(d))
            button.grid(row=row, column=col, padx=30, pady=30)

        # This creates the log frame in the bottom right
        self.log_frame = tk.Frame(self.root)
        self.log_frame.grid(row=1, column=1, padx=10, pady=10)

        self.log_label = Label(self.log_frame, text="Action Log:", font=('Helvetica', 10, 'bold'))
        self.log_label.grid(row=0, column=0)

        self.log_text = tk.Text(self.log_frame, width=40, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=1, column=0)

        self.running = False

    def process_frame(self, frame):
        """
        Processes the frame for display.

        Args:
            frame: The current frame from the video.

        Returns:
            imgtk: Tkinter compatible image to display.
        """
        # Here, this resizes the frame to match the display size and converts it to RGB
        frame = cv2.resize(frame, (self.display_width, self.display_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def lane_detection(self, frame):
        """
        Performs lane detection on the provided frame.

        Args:
            frame: The current frame from the video.

        Returns:
            frame_resized: The processed frame with lane detection applied.
        """
        # This resizes the frame to match the display size
        pts1 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
        pts2 = np.float32([[10, 10], [90, 5], [5, 90], [95, 95]])
        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

        binary = np.zeros((100, 100), dtype=np.uint8)
        skeleton = skeletonize(binary)  # This skeletonizes the binary image

        global left_fit_avg, right_fit_avg
        frame_resized = cv2.resize(frame, (self.display_width, self.display_height))

        # This loads the arrow image with transparency
        arrow_image = cv2.imread("C:\\Users\\baort\\Downloads\\up-arrow-12.png", cv2.IMREAD_UNCHANGED)
        arrow_width = 100
        arrow_height = int(arrow_image.shape[0] * (arrow_width / arrow_image.shape[1]))
        arrow_image = cv2.resize(arrow_image, (arrow_width, arrow_height))

        # This checks if the image has an alpha channel (transparency)
        if arrow_image.shape[2] == 4:  # If it has an alpha channel
            alpha_channel = arrow_image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
            bgr_image = arrow_image[:, :, :3]
        else:  # If no alpha channel, assume it is fully opaque
            alpha_channel = np.ones(arrow_image.shape[:2], dtype=np.float32)  # Fully opaque
            bgr_image = arrow_image

        # This defines the region of interest (ROI) for the arrow overlay
        height, width = frame_resized.shape[:2]
        y_offset = 10
        x_offset = width - arrow_width 
        roi = frame_resized[y_offset:y_offset + arrow_height, x_offset:x_offset + arrow_width]

        # Step 1: This clears the previous arrow by drawing a black rectangle over the previous position
        frame_resized[y_offset:y_offset + arrow_height, x_offset:x_offset + arrow_width] = (90)  # Black (or use your background color)

        # Step 2: This overlays the new arrow image onto the frame using the alpha mask
        for c in range(0, 3):  # Iterate over each color channel (BGR)
            frame_resized[y_offset:y_offset + arrow_height, x_offset:x_offset + arrow_width, c] = \
                (alpha_channel * bgr_image[:, :, c] + (1 - alpha_channel) * roi[:, :, c])


        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  
        # Converts the frame to grayscale for edge detection.
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  
        # Applies Gaussian blur to reduce noise.
        
        edges = cv2.Canny(blur, 50, 150)  
        # Performs Canny edge detection to highlight edges.

        mask = np.zeros_like(edges)  
        # Creates a mask for the region of interest.
        
        polygon = np.array([[
            (int(0.15 * width), int(height)),
            (int(0), int(height)),
            (int(0), int(0.75 * height)),
            (int(0.15 * width), int(0.70 * height)),
            (int(0.45 * width), int(0.55 * height)),
            (int(0.55 * width), int(0.55 * height)),
            (int(0.85 * width), int(0.75 * height)),
            (int(width), int(0.9 * height)),
            (int(width), int(height)),
        ]], np.int32)
        # Defines a polygon for the region of interest.

        cv2.fillPoly(mask, polygon, 255)  
        # Fills the region of interest in the mask.

        masked_edges = cv2.bitwise_and(edges, mask)  
        # Applies the mask to the edges image.

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)  
        # Detects lane lines using Hough Line Transform.

        left_lines, right_lines = [], []  
        # Initializes lists for left and right lane lines.

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                # Calculates the slope of the line.

                if -0.8 < slope < -0.4:
                    left_lines.append((x1, y1, x2, y2))  
                elif 0.4 < slope < 0.8:
                    right_lines.append((x1, y1, x2, y2))  

        def average_lane(lines, prev_fit):
            if len(lines) > 0:
                x_coords = [x1 for x1, _, x2, _ in lines] + [x2 for _, _, x2, _ in lines]
                y_coords = [y1 for _, y1, x2, _ in lines] + [y2 for _, _, _, y2 in lines]
                poly_fit = np.polyfit(x_coords, y_coords, 1)  

                if prev_fit is None:
                    return poly_fit  
                else:
                    return alpha * poly_fit + (1 - alpha) * prev_fit  

            return prev_fit  

        left_fit_avg = average_lane(left_lines, left_fit_avg)  
        right_fit_avg = average_lane(right_lines, right_fit_avg)  

        line_image = np.zeros_like(frame_resized)  
        # Creates an empty image for lane lines.

        def draw_lane_line(img, fit, color):
            if fit is not None:
                y1, y2 = img.shape[0], int(0.65 * img.shape[0])  
                x1 = int((y1 - fit[1]) / fit[0])  
                x2 = int((y2 - fit[1]) / fit[0])  
                cv2.line(img, (x1, y1), (x2, y2), color, 8)  

        draw_lane_line(line_image, left_fit_avg, (255, 0, 0))  
        draw_lane_line(line_image, right_fit_avg, (0, 255, 0))  

        if left_fit_avg is not None and right_fit_avg is not None:
            y1, y2 = frame_resized.shape[0], int(0.65 * frame_resized.shape[0])  
            x1_left = int((y1 - left_fit_avg[1]) / left_fit_avg[0])  
            x2_left = int((y2 - left_fit_avg[1]) / left_fit_avg[0])  
            x1_right = int((y1 - right_fit_avg[1]) / right_fit_avg[0])  
            x2_right = int((y2 - right_fit_avg[1]) / right_fit_avg[0])  

            x_center1 = (x1_left + x1_right) // 2  
            x_center2 = (x2_left + x2_right) // 2  

            cv2.line(line_image, (x_center1, y1), (x_center2, y2), (255, 255, 255), 8)  

        result = cv2.addWeighted(frame_resized, 0.8, line_image, 1, 0)  
        # This overlays the lane lines onto the original frame.

        cv2.rectangle(result, (x_offset, y_offset), (x_offset + arrow_width, y_offset + arrow_height), (27,26,25), -1)  
        # This clears the previous arrow.

        if left_fit_avg is not None and right_fit_avg is not None:
            center_slope = (x_center2 - x_center1) / (y2 - y1)  # This calculates the slope of the line
            angle = np.arctan(center_slope) * 180 / np.pi  # This takes the arctangent of the slope to find the angle of the line

            image_center = (arrow_width // 2, arrow_height // 2)  
            rotation_matrix = cv2.getRotationMatrix2D(image_center, 90, 1.0)  

            def rotate_image(image, angle):
                (h, w) = image.shape[:2]  
                center = (w // 2, h // 2)  
                M = cv2.getRotationMatrix2D(center, angle, 1.0)  
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))  # This rotates the arrow to whichever side needed
                return rotated  

            if angle > 15: # If the angle is greater than 15 degrees, then the arrow is turned right.
                arrow_image_rotated = rotate_image(arrow_image, 90)  
            elif angle < -15: # If the angle is less than -15 degrees, then the arrow is turned left.
                arrow_image_rotated = rotate_image(arrow_image, -90)  
            else:
                arrow_image_rotated = arrow_image  

            bgr_arrow = arrow_image_rotated[:, :, :3]  
            alpha_mask = arrow_image_rotated[:, :, 3] / 255.0  

            roi = result[y_offset:y_offset+arrow_height, x_offset:x_offset+arrow_width]  

            for c in range(3):  
                roi[:, :, c] = (1 - alpha_mask) * roi[:, :, c] + alpha_mask * bgr_arrow[:, :, c]  

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  
        img = Image.fromarray(result_rgb)  
        imgtk = ImageTk.PhotoImage(image=img)  

        return imgtk  



        frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
        result = np.zeros_like(frame_resized)
        return self.process_frame(result)
        return frame
        pass

    def update_frame(self):
        """Update the frames every calculated wait_time."""
        if self.running:
            ret, frame = self.cap.read()  
            # This reads a new frame from the video capture.

            if ret:
                normal_frame = self.process_frame(frame)  
                # This processes the current frame for normal display.
                lane_frame = self.lane_detection(frame)  
                # This processes the current frame for lane detection.

                self.top_frame.imgtk = normal_frame  
                self.top_frame.configure(image=normal_frame)  
                # This updates the top frame with the processed normal frame.

                self.bottom_frame.imgtk = lane_frame  
                self.bottom_frame.configure(image=lane_frame)  
                # This updates the bottom frame with the processed lane frame.

                self.root.after(self.wait_time, self.update_frame)  
                # This schedules the next frame update based on wait_time.
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
                # This resets the video capture to the first frame if reading fails.

    def start_video(self):
        """Start video processing."""
        if not self.running:
            self.running = True  
            # This marks the video as running.
            self.update_frame()  
            # This starts the frame update loop.

    def stop_video(self):
        """Stop video processing."""
        self.running = False  
        # This marks the video as stopped.

    def button_action(self, direction):
        """Handle button presses for directions."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  
        # This generates a timestamp for the button action.
        log_message = f"{self.username} clicked '{direction}' at {timestamp}\n"  
        # This formats a log message with the user's action and timestamp.

        # This adds the log message to the text box.
        self.log_text.config(state=tk.NORMAL)  
        self.log_text.insert(tk.END, log_message)  
        self.log_text.config(state=tk.DISABLED)  

        if direction == "PLAY":
            self.start_video()  
            # This starts video processing when the PLAY button is clicked.
        elif direction == "STOP":
            self.stop_video()  
            # This stops video processing when the STOP button is clicked.
        else:
            print(f"Moving {direction}")  
            # This prints the direction when any other button is clicked.

if __name__ == "__main__":
    create_db()  # This ensures the database exists.
    login_window = tk.Tk()  
    app = LoginWindow(login_window, "C:\\Users\\baort\\Downloads\\PWP (1).MP4")  
    login_window.mainloop()  
    # This starts the Tkinter main event loop for the login window.








   
