import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
import os
from trackerv2 import Tracker
from datetime import datetime
import license_plate_db
import math
from traffic_light_analyzer import analyze_traffic_light_color

# Load the YOLO model
model = YOLO("yolov10s.pt")

# Load class names from coco.txt
try:
    with open("coco.txt", "r") as my_file:
        data = my_file.read()
        class_list = data.split("\n")
except FileNotFoundError:
    print("Error: coco.txt not found. Please make sure it's in the same directory.")
    class_list = []


vehicle_classes_names = ['car', 'truck', 'bus', 'motorcycle']
traffic_light_class_name = 'traffic light'

# Get the corresponding class indices from class_list
vehicle_classes_indices = []
for name in vehicle_classes_names:
    try:
        vehicle_classes_indices.append(class_list.index(name))
    except ValueError:
        print(f"Warning: Vehicle class '{name}' not found in coco.txt. It will not be detected.")

try:
    traffic_light_class_index = class_list.index(traffic_light_class_name)
    detect_traffic_lights_object = True
except ValueError:
    print(f"Warning: '{traffic_light_class_name}' class not found in coco.txt. Traffic light object detection disabled.")
    detect_traffic_lights_object = False
    traffic_light_class_index = -1 # Indicate traffic light class is not available

tracker = Tracker() # Initialize the tracker (ensure this is the updated Tracker class)

count = 0 # Frame counter

# Define the violation area polygon (adjust based on your video feed)
# You can use the mouse callback functionality (RGB window) to get these points
area = [(324, 313), (283, 374), (854, 392), (864, 322)] # Example coordinates

# Create directory for today's date for saving images
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# List to keep track of vehicle IDs that have already committed a violation and been processed
violation_committed_ids = []

# --- Database Connection Management ---
# Establish database connection ONCE before the main processing loop
try:
    conn, cursor = license_plate_db.create_database() # Ensure create_database connects as needed
    db_connected = True
    print("Database connection established.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    print("Proceeding without database connection. Violations will be detected but NOT recorded in the database.")
    db_connected = False
    conn, cursor = None, None # Ensure conn and cursor are None if connection failed

# Video Capture (Still using hardcoded path - make this configurable, maybe using sys.argv or a config file)
video_path = r'D:\Projects\projectsocrates\Red-Traffic-Light-Violation-System\VID_20250507_13423185.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit() # Exit the script if video cannot be opened


# Mouse callback function for getting coordinates (useful for defining the 'area')
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Display coordinates in the console when mouse moves over the window
        # print(f"Mouse coordinates: ({x}, {y})")
        pass # Can be left empty or used for real-time display

# Create a window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


while True:
    ret, frame = cap.read()
    count += 1 # Increment frame counter

    # Frame skipping - process every other frame (adjust as needed for performance vs accuracy)
    if count % 2 != 0:
        continue

    if not ret:
        # If video ends, optionally loop or break
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Uncomment to loop video
        # continue # Uncomment to loop video
        break # Break the loop when video finishes

    # Resize frame for consistent processing (adjust dimensions as needed)
    frame = cv2.resize(frame, (1020, 600))
    display_frame = frame.copy() # Create a copy for drawing annotations

    # 1. Perform object detection (YOLO)
    # results is a list of Results objects, one per image (since we process one frame at a time)
    results = model(frame, verbose=False) # Set verbose=False to suppress prediction output

    # Extract detection data: [x1, y1, x2, y2, confidence, class_index]
    # results[0] is the Results object for the current frame
    if results and len(results[0].boxes) > 0:
         px = results[0].boxes.data
         # Convert to pandas DataFrame for easier filtering (optional but follows your pattern)
         # Ensure the tensor is on CPU before converting to numpy and then pandas
         px_cpu = px.cpu().numpy() if hasattr(px, 'is_cuda') and px.is_cuda else px.numpy()
         px_df = pd.DataFrame(px_cpu)
    else:
         # No detections in this frame, create an empty DataFrame
         px_df = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])


    # --- Traffic Light Detection and State Analysis ---
    # We want to determine the overall traffic light state relevant to the violation area.
    # For simplicity in this example, we'll assume if *any* detected traffic light
    # is RED, the overall state is RED. You might need more sophisticated logic
    # if multiple traffic lights are present or only certain ones are relevant.
    detected_traffic_light_state = "UNKNOWN" # Default state for the frame

    if detect_traffic_lights_object and traffic_light_class_index != -1 and not px_df.empty:
        # Filter YOLO results for traffic light detections based on the class index
        traffic_light_detections_df = px_df[px_df[5] == traffic_light_class_index].astype("float")

        # Process each detected traffic light object
        for index, row in traffic_light_detections_df.iterrows():
            x1, y1, x2, y2, confidence, class_index = row.tolist()

            # Ensure bounding box coordinates are valid integers and within frame bounds
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Check if the bounding box is valid (width and height are positive)
            if x2 > x1 and y2 > y1:
                # Crop the traffic light region of interest (ROI) from the original frame
                traffic_light_roi = frame[y1:y2, x1:x2]

                # Analyze the color within the ROI using the new function
                current_light_state = analyze_traffic_light_color(traffic_light_roi)

                # Update the overall traffic light state for the frame
                # Prioritize RED detection
                if current_light_state == "RED":
                    detected_traffic_light_state = "RED"
                    # Draw a red box and label on the display frame for RED traffic lights
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(display_frame, "RED LIGHT", (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                    # If we find a red light, we can assume the violation condition regarding the light is met
                    # for this frame and stop checking other traffic lights.
                    break
                elif current_light_state == "YELLOW" and detected_traffic_light_state != "RED":
                     # If not already red, check for yellow
                     detected_traffic_light_state = "YELLOW"
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box
                     cvzone.putTextRect(display_frame, "YELLOW LIGHT", (x1, y1 - 10), 1, 1, colorR=(0, 255, 255))
                elif current_light_state == "GREEN" and detected_traffic_light_state not in ["RED", "YELLOW"]:
                     # If not red or yellow, check for green
                     detected_traffic_light_state = "GREEN"
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                     cvzone.putTextRect(display_frame, "GREEN LIGHT", (x1, y1 - 10), 1, 1, colorR=(0, 255, 0))
                else:
                     # Draw grey box for detected traffic light with UNKNOWN state
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                     cvzone.putTextRect(display_frame, "TL UNKNOWN", (x1, y1 - 10), 1, 1, colorR=(128, 128, 128))


    # --- Process Vehicle Detections ---
    # Filter YOLO results to only include desired vehicle classes
    if vehicle_classes_indices and not px_df.empty:
        vehicle_detections_df = px_df[px_df[5].isin(vehicle_classes_indices)].astype("float")
    else:
        # If no vehicle classes found in coco.txt or no detections, create empty DataFrame
        vehicle_detections_df = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])


    # Prepare the list of detections for the tracker [x, y, w, h, class_name]
    objects_for_tracker = []
    for index, row in vehicle_detections_df.iterrows():
        x1, y1, x2, y2, confidence, class_index = row.tolist()
        class_name = class_list[int(class_index)]
        w, h = x2 - x1, y2 - y1
        # Append detection info in [x, y, w, h, class_name] format
        objects_for_tracker.append([int(x1), int(y1), int(w), int(h), class_name])

    # Update the tracker with the filtered vehicle detections
    # Assuming tracker.update is modified to accept [x, y, w, h, class_name] and frame count,
    # and returns a list of [x, y, w, h, id, class_name] for tracked objects in the current frame.
    tracked_objects_info = tracker.update(objects_for_tracker, count) # Pass current frame count

    # --- Process tracked objects for red light violation detection ---
    for obj_info in tracked_objects_info:
        x, y, w, h, id, obj_class_name = obj_info # Unpack tracked object information
        x1, y1, x2, y2 = x, y, x + w, y + h       # Get corner coordinates from [x, y, w, h]

        cx = int(x1 + x2) // 2 # Calculate center point X
        cy = int(y1 + y2) // 2 # Calculate center point Y

        # Check if the tracked object's center is within the violation area polygon
        # result will be positive if inside, negative if outside, 0 if on the boundary
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)

        # 6. Check for red light violation conditions
        # Condition: Object is a relevant vehicle type (already filtered by tracker input)
        # AND the object's center is within the violation area (result >= 0)
        # AND the overall detected traffic light state for the frame is "RED"
        # AND this vehicle ID has not already committed a violation in this sequence.
        if result >= 0: # Object is inside or on the boundary of the violation area
           if detected_traffic_light_state == "RED": # Check the overall detected traffic light state
                # Draw red box and ID for potential violators on the display frame
                cvzone.putTextRect(display_frame, f'ID: {id} ({obj_class_name})', (x1, y1 - 10), 1, 1, colorR=(0, 0, 255))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # If it's a new violation for this tracked ID (not in our list of committed violators)
                if id not in violation_committed_ids:
                   violation_committed_ids.append(id) # Add ID to committed violators list

                   # Save the original frame (without display annotations) for license plate processing
                   # Saving the original frame is better as annotations might interfere with LP detection.
                   timestamp = datetime.now().strftime('%H-%M-%S')
                   image_filename = f"{timestamp}_id{id}_{obj_class_name}.jpg" # Include ID and class in filename
                   output_path = os.path.join(output_dir, image_filename)
                   cv2.imwrite(output_path, frame) # Save the raw frame
                   print(f"Violation detected for ID {id} ({obj_class_name}) at frame {count}. Image saved: {output_path}")

                   # Process the saved image with license plate detection and add to database
                   if db_connected:
                       print(f"Processing license plate for ID {id} ({obj_class_name}) from saved image.")
                       # Call license_plate_db.process_single_image, passing conn and cursor
                       # process_single_image will handle the INSERT query
                       license_plate = license_plate_db.process_single_image(
                           output_path,
                           f"Red Light Violation ({obj_class_name})", # Include vehicle type in offense string
                           conn,    # Pass the connection object
                           cursor   # Pass the cursor object
                       )

                       if license_plate:
                           print(f"Detected License Plate for ID {id}: {license_plate}")
                           # Optional: Draw the detected LP on the displayed frame if you want to see it live
                           # This might add processing overhead and cause lag.
                           # cvzone.putTextRect(display_frame, f'LP: {license_plate}', (x1, y1 - 30), 1, 1, colorR=(0, 0, 255))
                       else:
                            print(f"No license plate detected for ID {id}.")
                   else:
                       print("Database not connected. Violation details not recorded.")

           else: # Object is a relevant vehicle in the area, but traffic light is NOT red
                # Draw green box and ID for tracked vehicles in the area that are NOT violating
                cvzone.putTextRect(display_frame, f'ID: {id}', (x1, y1 - 10), 1, 1, colorR=(0, 255, 0))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the violation area polygon on the display frame
    cv2.polylines(display_frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)

    # Display the current frame with all annotations
    cv2.imshow("RGB", display_frame) # Show the frame with annotations

    # Check for key press to exit
    # waitKey(1) displays the frame for 1 ms, allowing video playback
    # waitKey(0) would pause until a key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

# --- Cleanup ---
# Release the video capture object
cap.release()

# Close the database connection AFTER the main processing loop finishes
if db_connected and conn:
    conn.close()
    print("Database connection closed.")

# Destroy all OpenCV windows
cv2.destroyAllWindows()