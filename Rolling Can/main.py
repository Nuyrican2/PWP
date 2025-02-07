import cv2
import numpy as np


# Here, I resize the camera while maintaing the aspect ratio so that the frame looks the same on each computer.
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image, 1.0

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_image = cv2.resize(image, dim, interpolation=inter)
    return resized_image, r


# Function to detect the can
def detect_cans_in_video():
    cap = cv2.VideoCapture("/Users/pl1000790/Downloads/IMG_3876.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Here, I call the resize frame function so that the circle is detected on this frame.
        resized_frame, scale_factor = resize_with_aspect_ratio(frame, width=200)

        # Here, I gray and blur to create a reduction in noise, and more accurate detection.
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Here, I detect the circles with Hough Circles.
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=100,
            param2=60,
            minRadius=0,
            maxRadius=0
        )

        overlay_frame = frame.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))

            for circle in circles:
                x, y, r = circle
                # Because I detected the circles on a very small resized version of the image, i have to multiply by the reciprocal of the scale factor, so that when I display the original image, it shows the circle at normal size.
                x = int(x * (1 / scale_factor))
                y = int(y * (1 / scale_factor))
                r = int(r * (1 / scale_factor))

                # Here, I draw the circle and center.
                cv2.circle(overlay_frame, (x, y), r, (0, 255, 0), 3)
               
                cv2.circle(overlay_frame, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("Original Video with Circle Detection", overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            Break

    cap.release()
    cv2.destroyAllWindows()


# Start the program
detect_cans_in_video()