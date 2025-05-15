import cv2
import numpy as np

def analyze_traffic_light_color(traffic_light_roi):
    if traffic_light_roi is None or traffic_light_roi.size == 0:
        return "UNKNOWN"

    # Convert the region to HSV color space for color detection
    hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for Red, Yellow, Green signals
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create masks for each color within the ROI
    mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2) # Combine red masks

    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

    # Check for the presence of each color signal within the ROI.
    # We can count the number of non-zero pixels in the masks.
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    total_roi_pixels = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
    min_signal_pixels_threshold = total_roi_pixels * 0.03

    detected_state = "UNKNOWN"

    # Check for colors in priority order (e.g., Red > Yellow > Green)
    # A color is considered detected if its pixel count exceeds the threshold
    if red_pixels > min_signal_pixels_threshold:
        detected_state = "RED"
    elif yellow_pixels > min_signal_pixels_threshold:
        detected_state = "YELLOW"
    elif green_pixels > min_signal_pixels_threshold:
        detected_state = "GREEN"


    return detected_state