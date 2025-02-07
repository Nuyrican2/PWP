import cv2
import numpy as np

# Define the region of interest (ROI)
ROI_TOP = 130
ROI_BOTTOM = 370
ROI_LEFT = 200
ROI_RIGHT = 500


def interpolate(a1, a2, poly_deg=5, n_points=100):
    """
    Interpolates two detected curves using polynomial fitting and computes their midpoint curve.
    
    Args:
        a1 (numpy.ndarray): First detected curve as an array of (x, y) coordinates.
        a2 (numpy.ndarray): Second detected curve as an array of (x, y) coordinates.
        poly_deg (int, optional): Degree of the polynomial for fitting. Defaults to 5.
        n_points (int, optional): Number of interpolated points. Defaults to 100.
    
    Returns:
        numpy.ndarray: Array of interpolated midpoint coordinates.
    """
    # This extracts the x-coordinate range for both curves
    min_a1_x, max_a1_x = min(a1[:, 0]), max(a1[:, 0])
    min_a2_x, max_a2_x = min(a2[:, 0]), max(a2[:, 0])

    # This generates evenly spaced x-values for interpolation
    new_a1_x = np.linspace(min_a1_x, max_a1_x, n_points)
    new_a2_x = np.linspace(min_a2_x, max_a2_x, n_points)
    
    # This fits polynomials to both curves
    a1_coefs = np.polyfit(a1[:, 0], a1[:, 1], poly_deg)
    a2_coefs = np.polyfit(a2[:, 0], a2[:, 1], poly_deg)
    
    # This evaluates the polynomial to get the smoothed y-values
    new_a1_y = np.polyval(a1_coefs, new_a1_x)
    new_a2_y = np.polyval(a2_coefs, new_a2_x)
    
    # This calculates the midpoint of both curves
    midx = [np.mean([new_a1_x[i], new_a2_x[i]]) for i in range(n_points)]
    midy = [np.mean([new_a1_y[i], new_a2_y[i]]) for i in range(n_points)]
    
    return np.array([[int(x), int(y)] for x, y in zip(midx, midy)])


def detect_curves(frame):
    """
    Detects two dominant curves within a defined region of interest (ROI) in the image.
    
    Args:
        frame (numpy.ndarray): Input image frame from the video feed.
    
    Returns:
        tuple: Three numpy arrays representing the detected curves and their midpoint curve.
    """
    # This extracts the region of interest (ROI) from the frame
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
    
    # This converts the ROI to grayscale for edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # This applies Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # This detects edges using the Canny method
    edges = cv2.Canny(blurred, 50, 150)
    
    # This finds contours from the detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # This sorts the contours by area and selects the two largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # This ensures two curves are detected before proceeding
    if len(contours) < 2:
        return None, None, None
    
    # This converts contours into arrays of coordinates
    curve1 = contours[0].reshape(-1, 2)
    curve2 = contours[1].reshape(-1, 2)
    
    # This applies interpolation to smooth the curves and get their midpoint
    midpoints = interpolate(curve1, curve2)
    
    # This adjusts the coordinates relative to the original frame
    curve1 = curve1 + [ROI_LEFT, ROI_TOP]
    curve2 = curve2 + [ROI_LEFT, ROI_TOP]
    midpoints = midpoints + [ROI_LEFT, ROI_TOP]
    
    return curve1, curve2, midpoints


def main():
    """
    Captures video from the webcam and processes each frame to detect curves and their midpoint.
    """
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # This detects the two main curves and their midpoint
        curve1, curve2, midpoints = detect_curves(frame)
        
        if curve1 is not None and curve2 is not None and midpoints is not None:
            # This draws the first detected curve in blue
            for i in range(len(curve1) - 1):
                cv2.line(frame, tuple(curve1[i]), tuple(curve1[i + 1]), (255, 0, 0), 2)
            
            # This draws the second detected curve in blue
            for i in range(len(curve2) - 1):
                cv2.line(frame, tuple(curve2[i]), tuple(curve2[i + 1]), (255, 0, 0), 2)
            
            # This draws the interpolated midpoint curve in red
            for i in range(len(midpoints) - 1):
                cv2.line(frame, tuple(midpoints[i]), tuple(midpoints[i + 1]), (0, 0, 255), 2)
        
        # This draws the ROI rectangle for visualization
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 255, 0), 2)
        
        # This displays the processed frame
        cv2.imshow("Curve Detection with Smoothed Midpoint Line", frame)
        
        # This breaks the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
