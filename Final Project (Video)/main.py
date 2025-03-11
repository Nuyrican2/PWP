import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Global variables for lane smoothing
left_fit_avg = None
right_fit_avg = None
alpha = 0.1  # Smoothing factor

class LaneDetectionApp:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Lane Detection GUI")
        
        self.cap = cv2.VideoCapture(video_path)
        self.display_width = 1083
        self.display_height = 500
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Create frames for video feeds
        self.top_frame = Label(self.root)
        self.top_frame.pack()
        self.bottom_frame = Label(self.root)
        self.bottom_frame.pack()
        
        self.start_button = Button(self.root, text="Start", command=self.start_video)
        self.start_button.pack()
        
        self.stop_button = Button(self.root, text="Stop", command=self.stop_video)
        self.stop_button.pack()
        
        self.running = False
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.display_width, self.display_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk
    
    def lane_detection(self, frame):
        # Resize the frame to match the display size
        global left_fit_avg, right_fit_avg
        frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
        
        # Load the arrow image and resize it
        arrow_image = cv2.imread(r"C:\\Users\\baort\\Downloads\\up-arrow-12.png", cv2.IMREAD_UNCHANGED)
        arrow_width = 100
        arrow_height = int(arrow_image.shape[0] * (arrow_width / arrow_image.shape[1]))
        arrow_image = cv2.resize(arrow_image, (arrow_width, arrow_height))

        if arrow_image.shape[2] == 4:  # Handle alpha channel if present
            alpha_channel = arrow_image[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
            bgr_image = arrow_image[:, :, :3]
        else:
            alpha_channel = np.ones(arrow_image.shape[:2], dtype=np.float32)  # Fully opaque
            bgr_image = arrow_image

        # Region of Interest for the arrow overlay
        height, width = frame_resized.shape[:2]
        y_offset = 10
        x_offset = width - arrow_width - 10
        roi = frame_resized[y_offset:y_offset + arrow_height, x_offset:x_offset + arrow_width]
        roi[:, :] = (28,27,26)
        # Create an overlay with the arrow image
        for c in range(3):  # Loop over the color channels
            roi[:, :, c] = (1 - alpha_channel) * roi[:, :, c] + alpha_channel * bgr_image[:, :, c]

        # Proceed with the rest of the processing for lane detection...
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Region of Interest mask
        mask = np.zeros_like(edges)
        polygon = np.array([[ 
            (int(0.15 * width), int(height)),
            (int(0), int(height)),
            (int(0), int(0.75 * height)),
            (int(0.15 * width), int(0.70 * height)),
            (int(0.45 * width), int(0.55 * height)),
            (int(0.55 * width), int(0.55 * height)),
            (int(0.85 * width), int(0.75 * height)),
            (int(width), int(0.9*height)),
            (int(width), int(height)),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Line Transform to detect lane lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                if -0.8 < slope < -0.4:
                    left_lines.append((x1, y1, x2, y2))
                elif 0.4 < slope < 0.8:
                    right_lines.append((x1, y1, x2, y2))

        # Compute lane averages for stability
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

        # Create a blank image to draw lanes
        line_image = np.zeros_like(frame_resized)

        def draw_lane_line(img, fit, color):
            if fit is not None:
                y1, y2 = img.shape[0], int(0.65 * img.shape[0])
                x1 = int((y1 - fit[1]) / fit[0])
                x2 = int((y2 - fit[1]) / fit[0])
                cv2.line(img, (x1, y1), (x2, y2), color, 8)

        draw_lane_line(line_image, left_fit_avg, (255, 0, 0))  # Blue for left lane
        draw_lane_line(line_image, right_fit_avg, (0, 255, 0))  # Green for right lane

        # Draw the centerline
        if left_fit_avg is not None and right_fit_avg is not None:
            y1, y2 = frame_resized.shape[0], int(0.65 * frame_resized.shape[0])
            x1_left = int((y1 - left_fit_avg[1]) / left_fit_avg[0])
            x2_left = int((y2 - left_fit_avg[1]) / left_fit_avg[0])
            x1_right = int((y1 - right_fit_avg[1]) / right_fit_avg[0])
            x2_right = int((y2 - right_fit_avg[1]) / right_fit_avg[0])

            # Calculate the centerline (midpoint between left and right lines)
            x_center1 = (x1_left + x1_right) // 2
            x_center2 = (x2_left + x2_right) // 2

            # Draw the centerline
            cv2.line(line_image, (x_center1, y1), (x_center2, y2), (255, 255, 255), 8)

        # Overlay the detected lanes onto the original frame
        result = cv2.addWeighted(frame_resized, 0.8, line_image, 1, 0)

        # Convert the result to RGB for Tkinter display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        if left_fit_avg is not None and right_fit_avg is not None:
            # Calculate the slope of the centerline (midpoint between left and right lanes)
            center_slope = (x_center2 - x_center1) / (y2 - y1)

            # Calculate the angle with respect to the horizontal (bottom of the screen)
            angle = np.arctan(center_slope) * 180 / np.pi  # Convert to degrees

            # Determine if the car is turning left, right, or going straight
            if angle > 15:  # If angle is positive, car is turning left
                # Rotate the arrow 90 degrees clockwise
                arrow_image_rotated = cv2.rotate(arrow_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle < -15:  # If angle is negative, car is turning right
                # Rotate the arrow 90 degrees counterclockwise
                arrow_image_rotated = cv2.rotate(arrow_image, cv2.ROTATE_90_CLOCKWISE)
            else:  # If angle is close to zero, car is going straight
                arrow_image_rotated = arrow_image

            # Overlay the rotated arrow image
            if arrow_image_rotated.shape[2] == 4:
                alpha_channel = arrow_image_rotated[:, :, 3]
                bgr_image = arrow_image_rotated[:, :, :3]
            else:
                alpha_channel = np.ones(arrow_image_rotated.shape[:2], dtype=np.uint8) * 255
                bgr_image = arrow_image_rotated

            mask = alpha_channel / 255.0
            roi = result[y_offset:y_offset + arrow_height, x_offset:x_offset + arrow_width]
            for c in range(3):
                roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * bgr_image[:, :, c]
        
        


        # Convert the result to ImageTk format for Tkinter
        img = Image.fromarray(result_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk


    def update_frame(self):
        """Update the frames every 10ms."""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                normal_frame = self.process_frame(frame)
                lane_frame = self.lane_detection(frame)
                
                self.top_frame.imgtk = normal_frame
                self.top_frame.configure(image=normal_frame)
                
                self.bottom_frame.imgtk = lane_frame
                self.bottom_frame.configure(image=lane_frame)
                
                self.root.after(10, self.update_frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def start_video(self):
        """Start video processing."""
        if not self.running:
            self.running = True
            self.update_frame()
    
    def stop_video(self):
        """Stop video processing."""
        self.running = False
        self.top_frame.configure(image='')
        self.bottom_frame.configure(image='')

if __name__ == "__main__":
    video_path = r"C:\\Users\\baort\\Downloads\\PWP (1).mp4"
    root = tk.Tk()
    app = LaneDetectionApp(root, video_path)
    root.mainloop()
