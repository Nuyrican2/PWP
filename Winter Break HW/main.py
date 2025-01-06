import cv2
import numpy as np
import math

def filter_lines(lines, threshold=100):
    filtered_lines = []
    midpoints = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Here, I check if the midpoint is at least 100 pixels away from all other midpoints.
        if all(math.sqrt((midpoint[0] - existing_mp[0])**2 + (midpoint[1] - existing_mp[1])**2) >= threshold for existing_mp in midpoints):
            filtered_lines.append(line[0])
            midpoints.append(midpoint)

    return filtered_lines

# This function calcluates slope taking the rise over run of the x and y coordinates.
def calculate_slope(line):
    x1, y1, x2, y2 = line
    # If the slope is vertical, it returns 'inf' standing for infinite, since there would otherwise be a divide by 0 error.
    if x2 - x1 == 0:  
        return float('inf')
    return (y2 - y1) / (x2 - x1)

# I created this midpoint function to be used later in other parts of the program.
def get_midpoint(line):

    x1, y1, x2, y2 = line
    return (x1 + x2) / 2



def draw_centerline(frame, lines):
    # If there are less than two lines, nothing takes place.
    if len(lines) < 2:
        return

    # Here, I sorted lines by x-coordinate of the midpoints using the get_midpoint function.
    lines.sort(key=get_midpoint)
    line1, line2 = lines[:2]

    # I calculate the slopes of the lines here.
    slope1 = calculate_slope(line1)
    slope2 = calculate_slope(line2)

    # Here, I'm doing some basic error blocking, so that when the center line is drawn, it is either drawn horizontally or vertically.
    if abs(slope1) < 1.5:  # If the absolute value of the slope is less than 1.5 (Slightly greater than diagonal), then I will assume it's being held horizontally.
        y_center = int((line1[1] + line1[3] + line2[1] + line2[3]) / 4)
        cv2.line(frame, (0, y_center), (frame.shape[1], y_center), (0, 0, 255), 2)
    else:  # Otherwise, I can assume the lines are vertical.
        x_center = int((line1[0] + line1[2] + line2[0] + line2[2]) / 4)
        cv2.line(frame, (x_center, 0), (x_center, frame.shape[0]), (0, 0, 255), 2)

def main():
    # Here, I start the camera and video stream.
    camera = cv2.VideoCapture(0)



    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # The first step in creating the region of interest (ROI) is determing the dimensions of the frame, which I do here.
        height, width = frame.shape[:2]

        # Here, I define the size of the ROI, to be half of the length of the minimum of the width or height of the frame.
        roi_size = min(width, height) // 2   
        x = (width - roi_size) // 2  # Here, I define the x-coordinate for the square to be in the middle.
        y = (height - roi_size) // 2  # I do the same for the y-coordinate here as well.

        # Here, for efficiency when detecting edges, I both gray the image, and blur it.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Here, I detected the edges in the image for line detection with the built in canny edge detection function.
        edges = cv2.Canny(blurred, 50, 150)

        # Here, I just made a black image so that I could build the rest of the display off of it.
        black_image = np.zeros_like(edges)  # This is the black image.
        black_image[y:y+roi_size, x:x+roi_size] = edges[y:y+roi_size, x:x+roi_size]  # This makes it fill to the screen based off of the size of the frame with the detected edges.

        # Now that I have an ROI demarcated, I specifically do Hough Line detection within that region.
        lines = cv2.HoughLinesP(black_image, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        if lines is not None:
            # Here I ensure that if there are multiple lines close together, only one representative line is kept. (Using the function I defined in the beginning)
            filtered_lines = filter_lines(lines)

            # I draw the lines I filtered out in the color green.
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Here, I draw the red centerline in between the two detected lines.
            draw_centerline(frame, filtered_lines)

        # Here, I created a white background, so that no excess material would be detected.
        white_background = np.ones_like(frame) * 255

        # Here, I copied the already created ROI frame ON to the white background.
        roi_frame = frame[y:y+roi_size, x:x+roi_size]
        white_background[y:y+roi_size, x:x+roi_size] = roi_frame  # Here, I essentailly paste it.

        # I drew a green rectangle around the ROI as an outline.
        cv2.rectangle(white_background, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2) 

        # Here, I crop the image to only show the ROI for visual aesthetics.
        final_frame = white_background[y:y+roi_size, x:x+roi_size]  

        # Here I display the frame, indicating the dimensions to be that of the final frame (ROI only)
        cv2.imshow('Lane Detection', final_frame)

        # Here, I make it so that I can quit the application by pressing the 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
