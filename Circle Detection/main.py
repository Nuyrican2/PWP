import cv2
import numpy as np

def detect_circles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    output = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=30, 
        maxRadius=100
    )

    if circles is not None:
        # Convert circle parameters to integers
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Create a mask for the detected circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, thickness=-1)

            # Extract the region of interest (ROI) using the mask
            roi = cv2.bitwise_and(gray, gray, mask=mask)

            # Find contours in the ROI
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Assume the largest contour corresponds to the detected circle
                contour = max(contours, key=cv2.contourArea)

                # Approximate the contour to check for vertices
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

                # If the contour has no vertices (e.g., is a circle), draw it
                if len(approx) > 8:  # More than 8 vertices usually indicates a circle
                    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                    cv2.putText(output, "Circle", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Detected Circles", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_circles("path_to_your_image.jpg")
