from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import fetch_car, extract_license_plate, save_to_csv

output_data = {}

vehicle_tracker = Sort()

# Load models with GPU
object_detection_model = YOLO('yolov8n.pt')  # Automatically uses GPU if available
license_plate_model = YOLO('LPD.pt')  # Automatically uses GPU if available


# Load video
video_capture = cv2.VideoCapture('./sample.mp4')

vehicle_classes = [2, 3, 5, 7]

# Read frames
frame_number = -1
has_frames = True
while has_frames:
    frame_number += 1
    has_frames, current_frame = video_capture.read()
    if has_frames:
        output_data[frame_number] = {}
        # Detect vehicles
        detected_objects = object_detection_model(current_frame)[0]
        filtered_detections = []
        for obj in detected_objects.boxes.data.tolist():
            x1, y1, x2, y2, confidence, category_id = obj
            if int(category_id) in vehicle_classes:
                filtered_detections.append([x1, y1, x2, y2, confidence])

        # Track vehicles
        tracked_vehicle_ids = vehicle_tracker.update(np.asarray(filtered_detections))

        # Detect license plates
        detected_license_plates = license_plate_model(current_frame)[0]
        for plate in detected_license_plates.boxes.data.tolist():
            x1, y1, x2, y2, confidence, category_id = plate

            # Assign license plate to car
            car_x1, car_y1, car_x2, car_y2, vehicle_id = fetch_car(plate, tracked_vehicle_ids)

            if vehicle_id != -1:
                # Crop license plate
                license_plate_image = current_frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                gray_plate = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
                _, thresholded_plate = cv2.threshold(gray_plate, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                plate_text, plate_text_confidence = extract_license_plate(thresholded_plate)

                if plate_text is not None:
                    output_data[frame_number][vehicle_id] = {
                        'car': {'bounding_box': [car_x1, car_y1, car_x2, car_y2]},
                        'license_plate': {
                            'bounding_box': [x1, y1, x2, y2],
                            'text': plate_text,
                            'bbox_confidence': confidence,
                            'text_confidence': plate_text_confidence
                        }
                    }

# Save results
save_to_csv(output_data, './test.csv')