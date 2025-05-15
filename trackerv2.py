# Modified tracker.py

import math

class Tracker:
    def __init__(self):
        # Dictionary to store information about tracked objects.
        # Key: object ID (int)
        # Value: Dictionary with keys:
        # - 'center': tuple (cx, cy) of the object's center in the last seen frame
        # - 'class': string, the class name of the object (e.g., 'car', 'truck')
        # - 'bbox': list [x, y, w, h] of the bounding box in the last seen frame
        # - 'last_seen_frame': int, the frame number the object was last detected
        self.tracked_objects = {}

        # Counter for assigning new object IDs. Starts from 0.
        self.id_count = 0

        # --- Configuration for cleanup ---
        # Maximum number of frames to keep a tracked object if it's not detected.
        # If an object is lost for more than this many frames, it's removed.
        # Adjust this value based on your video's frame rate and expected object behavior (e.g., brief occlusions).
        self.max_frames_lost = 15 # Example: Keep lost objects for 15 frames

        # --- Configuration for matching ---
        # Maximum distance (in pixels) between a new detection's center and a tracked object's center
        # for them to be considered the same object.
        # Adjust this based on your video resolution and how fast objects move.
        self.match_distance_threshold = 50 # Example: 50 pixels


    # Modified update method to accept objects with class and the current frame number
    def update(self, objects_info_list, current_frame_number):
        """
        Updates the tracker with the current frame's object detections.
        Matches current detections to existing tracked objects, updates their info,
        assigns new IDs to unmatched detections, and cleans up lost objects.

        Args:
            objects_info_list: A list of lists, where each inner list contains
                               [x, y, w, h, class_name] for a detected object
                               in the current frame.
            current_frame_number: The frame number currently being processed (int).

        Returns:
            A list of lists, where each inner list contains
            [x, y, w, h, id, class_name] for the tracked objects that were
            found and updated in the current frame.
            Returns an empty list if no objects are tracked in this frame.
        """
        # List to store the information of tracked objects found in the current frame.
        # This list will be returned at the end of the function.
        objects_in_current_frame = []

        # Keep track of which current detections have been matched to a tracked object.
        current_detections_matched_indices = set()

        # Keep track of which tracked object IDs have been updated in this frame.
        tracked_ids_updated_in_frame = set()


        # --- Matching Step ---
        # Iterate through current frame detections and try to match them with existing tracked objects.
        # Using a nested loop (current detections vs. tracked objects) for matching.
        # Store potential matches and select the best one (closest distance) if multiple tracked objects are close.
        potential_matches = {} # Key: index of current detection, Value: list of (distance, tracked_id) tuples

        for i, obj_info in enumerate(objects_info_list):
             # Ensure obj_info has the expected structure [x, y, w, h, class_name]
             if len(obj_info) != 5:
                 print(f"Warning: Unexpected object_info format: {obj_info}. Skipping.")
                 continue # Skip this detection if format is wrong

             x, y, w, h, class_name = obj_info
             # Ensure bounding box dimensions are valid before calculating center
             if w <= 0 or h <= 0:
                 # print(f"Warning: Invalid bounding box dimensions: w={w}, h={h}. Skipping detection {obj_info}.")
                 continue # Skip detection with invalid dimensions

             cx = int((x + x + w) / 2) # Calculate center x
             cy = int((y + y + h) / 2) # Calculate center y

             potential_matches[i] = [] # Initialize list of potential matches for this detection

             # Iterate through currently tracked objects to find a match
             # Use list() to iterate over a copy of keys, in case self.tracked_objects changes (though it shouldn't in this loop)
             for id in list(self.tracked_objects.keys()):
                 tracked_obj_data = self.tracked_objects[id]
                 tracked_center = tracked_obj_data.get('center') # Use .get() for safety

                 # Ensure tracked object data is valid
                 if tracked_center is None or len(tracked_center) != 2:
                     # print(f"Warning: Invalid tracked object data for ID {id}. Removing.")
                     del self.tracked_objects[id] # Remove invalid entry
                     continue

                 # Calculate distance between current detection center and tracked object center
                 dist = math.hypot(cx - tracked_center[0], cy - tracked_center[1])

                 # If the distance is within the threshold, consider it a potential match
                 if dist < self.match_distance_threshold:
                     potential_matches[i].append((dist, id))

        # Resolve potential matches: Assign each current detection to the closest available tracked object.
        # This greedy approach works reasonably well for simple cases.
        # Sort potential matches by distance to prioritize closer matches.
        sorted_potential_matches = []
        for det_index, matches in potential_matches.items():
            for dist, tracked_id in matches:
                sorted_potential_matches.append((dist, det_index, tracked_id))

        sorted_potential_matches.sort() # Sort by distance (ascending)

        assigned_detections = set() # Indices of current detections that have been assigned to a tracked object
        assigned_tracked_ids = set() # IDs of tracked objects that have been matched

        for dist, det_index, tracked_id in sorted_potential_matches:
             # Check if both the current detection and the tracked object have not been assigned yet
             if det_index not in assigned_detections and tracked_id not in assigned_tracked_ids:
                  # This is a valid, unassigned match. Assign the detection to the tracked object.
                  x, y, w, h, class_name = objects_info_list[det_index]
                  cx = int((x + x + w) / 2)
                  cy = int((y + y + h) / 2)

                  # Update the tracked object's information with the current frame's detection data
                  self.tracked_objects[tracked_id] = {
                      'center': (cx, cy),
                      'class': class_name,
                      'bbox': [x, y, w, h],
                      'last_seen_frame': current_frame_number
                  }
                  # Add the updated tracked object to the list of objects in the current frame
                  objects_in_current_frame.append([x, y, w, h, tracked_id, class_name])

                  # Mark the detection and tracked ID as assigned
                  assigned_detections.add(det_index)
                  assigned_tracked_ids.add(tracked_id)

        # --- Handling Unmatched Detections (New Objects) ---
        # Iterate through current detections again. If a detection was not assigned (is new), create a new tracked object for it.
        for i, obj_info in enumerate(objects_info_list):
            # Check if the detection was not assigned during the matching step
            if i not in assigned_detections:
                 # This detection is a new object. Assign a new ID.
                 x, y, w, h, class_name = obj_info
                 cx = int((x + x + w) / 2)
                 cy = int((y + y + h) / 2)

                 new_id = self.id_count
                 self.tracked_objects[new_id] = {
                      'center': (cx, cy),
                      'class': class_name,
                      'bbox': [x, y, w, h],
                      'last_seen_frame': current_frame_number
                 }
                 objects_in_current_frame.append([x, y, w, h, new_id, class_name])
                 self.id_count += 1


        # --- Cleanup Step ---
        # Remove tracked objects that have not been seen for more than max_frames_lost frames.
        # Iterate through a copy of keys because we might delete items during iteration.
        ids_to_remove = [
            id for id in list(self.tracked_objects.keys())
            if current_frame_number - self.tracked_objects[id]['last_seen_frame'] > self.max_frames_lost
        ]

        for id in ids_to_remove:
            del self.tracked_objects[id]
            # print(f"Tracker: Removed object ID {id} due to inactivity.") # Optional: print cleanup info

        # Return the list of objects that are currently being tracked and were found in this frame.
        # This list contains [x, y, w, h, id, class_name] for each object found in the current frame.
        # If no objects were tracked in this frame, an empty list is returned.
        return objects_in_current_frame