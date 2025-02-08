import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(image, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(image, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(image, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(image, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(image, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(image, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(image, (x2, y1), (x2, y2), color, thickness)

    cv2.line(image, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(image, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return image


# Load CSV results
results_df = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = 'sample.mp4'
video_capture = cv2.VideoCapture(video_path)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# Extract license plate information
license_plate_data = {}
for vehicle_id in np.unique(results_df['car_id']):
    max_score = np.amax(results_df[results_df['car_id'] == vehicle_id]['license_number_score'])
    license_plate_data[vehicle_id] = {
        'license_crop': None,
        'license_plate_number': results_df[(results_df['car_id'] == vehicle_id) &
                                           (results_df['license_number_score'] == max_score)]['license_number'].iloc[0]
    }
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, results_df[(results_df['car_id'] == vehicle_id) &
                                                          (results_df['license_number_score'] == max_score)]['frame_nmr'].iloc[0])
    ret, frame = video_capture.read()

    x1, y1, x2, y2 = ast.literal_eval(results_df[(results_df['car_id'] == vehicle_id) &
                                                 (results_df['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate_data[vehicle_id]['license_crop'] = license_crop


frame_number = -1

video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames
ret = True
while ret:
    ret, frame = video_capture.read()
    frame_number += 1
    if ret:
        frame_data = results_df[results_df['frame_nmr'] == frame_number]
        for row_index in range(len(frame_data)):
            # Draw car bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(frame_data.iloc[row_index]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw license plate bounding box
            x1, y1, x2, y2 = ast.literal_eval(frame_data.iloc[row_index]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Crop and overlay license plate
            license_crop = license_plate_data[frame_data.iloc[row_index]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate_data[frame_data.iloc[row_index]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate_data[frame_data.iloc[row_index]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        video_writer.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

video_writer.release()
video_capture.release()