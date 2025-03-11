import cv2
import numpy as np


# ---------------------------
# ROI settings
# ---------------------------
ROI_TOP = 130
ROI_BOTTOM = 370
ROI_LEFT = 200
ROI_RIGHT = 500


# Global smoothing parameters
smoothed_midpoints = None
SMOOTHING_ALPHA = 0.05  # Lower value gives smoother but slower response


# Fixed number of midpoints for consistent smoothing
NUM_POINTS = 50


# Global variable to store Kalman filters (one per midline point)
kalman_filters = None


# ---------------------------
# Compute Midline Directly from Contours
# ---------------------------
def compute_midline(curve1, curve2):
   # Ensure both curves have the same number of points
   min_len = min(len(curve1), len(curve2))
   curve1, curve2 = curve1[:min_len], curve2[:min_len]
   # Compute midpoints (using integer division)
   midpoints = (curve1 + curve2) // 2
   return midpoints


# ---------------------------
# Resample midpoints to a fixed number of points
# ---------------------------
def resample_midpoints(midpoints, num_points=NUM_POINTS):
   # Ensure midpoints is in float format
   midpoints = midpoints.astype(np.float32)
   if len(midpoints) == num_points:
       return midpoints
   indices = np.linspace(0, len(midpoints) - 1, num_points)
   resampled = np.empty((num_points, 2), dtype=np.float32)
   for i, idx in enumerate(indices):
       low = int(np.floor(idx))
       high = min(int(np.ceil(idx)), len(midpoints) - 1)
       weight = idx - low
       resampled[i] = (1 - weight) * midpoints[low] + weight * midpoints[high]
   return resampled


# ---------------------------
# Kalman Filter Setup for 2D points
# ---------------------------
def create_kalman_filter():
   # Here we use a state of 4: [x, y, dx, dy] and measurement of 2: [x, y]
   kalman = cv2.KalmanFilter(4, 2)
   kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)
   kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32)
   kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
   kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
   kalman.errorCovPost = np.eye(4, dtype=np.float32)
   return kalman


def kalman_smooth_midpoints(midpoints):
   global kalman_filters
   # Initialize Kalman filters if not already done or if the number of points has changed
   if kalman_filters is None or len(kalman_filters) != len(midpoints):
       kalman_filters = [create_kalman_filter() for _ in range(len(midpoints))]
       for i, point in enumerate(midpoints):
           # Initialize each filter's state with the current measurement.
           kalman_filters[i].statePre = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
           kalman_filters[i].statePost = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
   smoothed = []
   for i, point in enumerate(midpoints):
       measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
       kalman_filters[i].predict()
       estimated = kalman_filters[i].correct(measurement)
       smoothed.append([estimated[0, 0], estimated[1, 0]])
   return np.array(smoothed, dtype=np.float32)


# ---------------------------
# Detect Parallel Curves with Additional Preprocessing
# ---------------------------
def detect_curves(frame):
   # Extract the region of interest
   roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
   gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (9, 9), 0)
   edges = cv2.Canny(blurred, 50, 150)


   # Apply morphological closing to clean up the edges
   kernel = np.ones((3, 3), np.uint8)
   edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


   # Draw all detected contours on ROI for debugging purposes
   cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)


   # Find the two largest contours (assumed to be the parallel lines)
   valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]


   if len(valid_contours) < 2:
       return None, None, None, edges  # Not enough contours detected


   # Approximate the contours to reduce noise
   curve1 = cv2.approxPolyDP(valid_contours[0], epsilon=5, closed=False).reshape(-1, 2)
   curve2 = cv2.approxPolyDP(valid_contours[1], epsilon=5, closed=False).reshape(-1, 2)


   # Compute the midline from the two curves
   midpoints = compute_midline(curve1, curve2)


   # Adjust the coordinates from the ROI to the full frame
   curve1 += [ROI_LEFT, ROI_TOP]
   curve2 += [ROI_LEFT, ROI_TOP]
   midpoints += [ROI_LEFT, ROI_TOP]


   return curve1, curve2, midpoints, edges


# ---------------------------
# Draw Everything on the Frame
# ---------------------------
def draw_centerline_on_frame(frame, curve1, curve2, midpoints, edges):
   # Draw curve1 in red
   if curve1 is not None:
       for point in curve1:
           cv2.circle(frame, tuple(map(int, point)), 1, (0, 0, 255), -1)
   # Draw curve2 in green
   if curve2 is not None:
       for point in curve2:
           cv2.circle(frame, tuple(map(int, point)), 1, (0, 255, 0), -1)
   # Draw the centerline (midpoints connected by blue lines)
   if midpoints is not None:
       for i in range(len(midpoints) - 1):
           pt1 = tuple(map(int, midpoints[i]))
           pt2 = tuple(map(int, midpoints[i + 1]))
           cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


   # Overlay the processed edge image on the ROI
   if edges is not None:
       edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
       frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = cv2.addWeighted(
           frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT], 0.7, edges_colored, 0.3, 0
       )


   return frame


# ---------------------------
# Main Function for Live Video Processing
# ---------------------------
def main():
   global smoothed_midpoints
   cap = cv2.VideoCapture(0)


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       curve1, curve2, midpoints, edges = detect_curves(frame)


       if midpoints is not None:
           # Convert midpoints to float for smoothing
           current_midpoints = midpoints.astype(np.float32)
           # Resample midpoints to ensure a fixed number of points
           current_midpoints = resample_midpoints(current_midpoints, num_points=NUM_POINTS)


           if smoothed_midpoints is None or len(smoothed_midpoints) != len(current_midpoints):
               smoothed_midpoints = current_midpoints
           else:
               # Apply exponential moving average smoothing
               smoothed_midpoints = (SMOOTHING_ALPHA * current_midpoints +
                                     (1 - SMOOTHING_ALPHA) * smoothed_midpoints)
           # Further smooth with the Kalman filter
           smoothed_midpoints = kalman_smooth_midpoints(smoothed_midpoints)


           # Draw curves and the smoothed centerline
           frame = draw_centerline_on_frame(frame, curve1, curve2,
                                            smoothed_midpoints.astype(np.int32), edges)


       # Draw the ROI rectangle for visualization
       cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 255), 2)


       # Display the resulting frame
       cv2.imshow("Live Centerline Detection", frame)


       # Exit on pressing 'q'
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()
