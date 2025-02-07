import cv2
import numpy as np

# Function to resize the image while maintaining the aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image, 1.0  # Return the original image and scaling factor of 1.0

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_image = cv2.resize(image, dim, interpolation=inter)
    return resized_image, r  # Return resized image and scaling factor

# Function to detect cans in a live video feed
def detect_cans_in_video():
    cap = cv2.VideoCapture(0)  # Open the camera (0 is typically the default camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame for faster processing and get the scaling factor
        resized_frame, scale_factor = resize_with_aspect_ratio(frame, width=150)

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=100,
            param2=50,
            minRadius=0,
            maxRadius=0
        )

        # Prepare the original video frame for overlay
        overlay_frame = frame.copy()  # We'll draw on this frame

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))  # Round and convert to integers
            
            for circle in circles:
                x, y, r = circle
                # Scale the circle coordinates and radius back to the original image
                x = int(x * 1/scale_factor)
                y = int(y * 1/scale_factor)
                r = int(r * 1/scale_factor)

                # Draw the outer circle on the original frame (overlay)
                cv2.circle(overlay_frame, (x, y), r, (0, 255, 0), 3)
                # Draw the center of the circle on the original frame (overlay)
                cv2.circle(overlay_frame, (x, y), 2, (0, 0, 255), 3)

        # Display the original video frame with circle overlays
        cv2.imshow("Original Video with Circle Detection", overlay_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print the final scale factor at the end of the program
    print(f"Final scale factor: {scale_factor}")

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Start the can detection on the live video feed
detect_cans_in_video()
