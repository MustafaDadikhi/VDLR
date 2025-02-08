import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(input_data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in input_data])
    vehicle_ids = np.array([int(float(row['car_id'])) for row in input_data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in input_data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in input_data])

    interpolated_results = []
    unique_vehicle_ids = np.unique(vehicle_ids)
    for vehicle_id in unique_vehicle_ids:

        frame_numbers_ = [p['frame_nmr'] for p in input_data if int(float(p['car_id'])) == int(float(vehicle_id))]
        print(frame_numbers_, vehicle_id)

        # Filter data for a specific vehicle ID
        vehicle_mask = vehicle_ids == vehicle_id
        vehicle_frame_numbers = frame_numbers[vehicle_mask]
        interpolated_car_bboxes = []
        interpolated_license_plate_bboxes = []

        first_frame = vehicle_frame_numbers[0]
        last_frame = vehicle_frame_numbers[-1]

        for i in range(len(car_bboxes[vehicle_mask])):
            frame_number = vehicle_frame_numbers[i]
            car_bbox = car_bboxes[vehicle_mask][i]
            license_plate_bbox = license_plate_bboxes[vehicle_mask][i]

            if i > 0:
                prev_frame_number = vehicle_frame_numbers[i - 1]
                prev_car_bbox = interpolated_car_bboxes[-1]
                prev_license_plate_bbox = interpolated_license_plate_bboxes[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes_gap = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes_gap = interp_func(x_new)

                    interpolated_car_bboxes.extend(interpolated_car_bboxes_gap[1:])
                    interpolated_license_plate_bboxes.extend(interpolated_license_plate_bboxes_gap[1:])

            interpolated_car_bboxes.append(car_bbox)
            interpolated_license_plate_bboxes.append(license_plate_bbox)

        for i in range(len(interpolated_car_bboxes)):
            frame_number = first_frame + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(vehicle_id)
            row['car_bbox'] = ' '.join(map(str, interpolated_car_bboxes[i]))
            row['license_plate_bbox'] = ' '.join(map(str, interpolated_license_plate_bboxes[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in input_data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(vehicle_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_results.append(row)

    return interpolated_results


# Load the CSV file
with open('test.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    data = list(csv_reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=header)
    csv_writer.writeheader()
    csv_writer.writerows(interpolated_data)