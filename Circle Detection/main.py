import cv2
import numpy as np

def detect_circles_and_draw_contours(image_path):
    # Load the image
    image = cv2.imread(image_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50,
        param1=100, 
        param2=30, 
        minRadius=0, 
        maxRadius=0
    )

    # Placeholder for the main circle (can outline)
    main_circle = None

    # If circles are detected, sort them by radius and find the largest
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda c: c[2], reverse=True)  # Sort by radius
        main_circle = circles[0]  # Assume the largest circle is the main can outline

        # Draw the main circle
        cv2.circle(output, (main_circle[0], main_circle[1]), main_circle[2], (0, 255, 0), 4)  # Green outline
        cv2.circle(output, (main_circle[0], main_circle[1]), 2, (0, 0, 255), 3)  # Red center

    # Edge detection for contours (backup method)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours as fallback
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:  # Avoid division by zero
            continue
        
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Filter contours: circularity, area, and size
        if 0.7 < circularity <= 1.2 and 30 < radius < 100 and area > 500:
            # Exclude smaller circles fully contained within the main circle
            if main_circle is not None:
                dist_to_main = np.sqrt((main_circle[0] - center[0]) ** 2 + (main_circle[1] - center[1]) ** 2)
                if dist_to_main + radius < main_circle[2]:  # Fully contained
                    continue

            # Draw the contour circle
            cv2.circle(output, center, radius, (255, 0, 0), 2)  # Blue fallback circle

    # Show results
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Circles and Contours", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# File path to the image
image_path = "C:\\Users\\baort\\OneDrive\\Desktop\\top.jpg"

# Run the function
detect_circles_and_draw_contours(image_path)
