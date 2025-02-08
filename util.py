import string
import easyocr

# Initialize the OCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
char_to_int_mapping = {'O': '0',
                       'I': '1',
                       'J': '3',
                       'A': '4',
                       'G': '6',
                       'S': '5'}

int_to_char_mapping = {'0': 'O',
                       '1': 'I',
                       '3': 'J',
                       '4': 'A',
                       '6': 'G',
                       '5': 'S'}


def save_to_csv(output_data, output_file_path):
    """
    Write the results to a CSV file.

    Args:
        output_data (dict): Dictionary containing the results.
        output_file_path (str): Path to the output CSV file.
    """
    with open(output_file_path, 'w') as file:
        file.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'vehicle_id', 'car_bbox',
                                                   'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                   'license_number_score'))

        for frame_number in output_data.keys():
            for vehicle_id in output_data[frame_number].keys():
                print(output_data[frame_number][vehicle_id])
                if 'car' in output_data[frame_number][vehicle_id].keys() and \
                   'license_plate' in output_data[frame_number][vehicle_id].keys() and \
                   'text' in output_data[frame_number][vehicle_id]['license_plate'].keys():
                    file.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                               vehicle_id,
                                                               '[{} {} {} {}]'.format(
                                                                   output_data[frame_number][vehicle_id]['car']['bounding_box'][0],
                                                                   output_data[frame_number][vehicle_id]['car']['bounding_box'][1],
                                                                   output_data[frame_number][vehicle_id]['car']['bounding_box'][2],
                                                                   output_data[frame_number][vehicle_id]['car']['bounding_box'][3]),
                                                               '[{} {} {} {}]'.format(
                                                                   output_data[frame_number][vehicle_id]['license_plate']['bounding_box'][0],
                                                                   output_data[frame_number][vehicle_id]['license_plate']['bounding_box'][1],
                                                                   output_data[frame_number][vehicle_id]['license_plate']['bounding_box'][2],
                                                                   output_data[frame_number][vehicle_id]['license_plate']['bounding_box'][3]),
                                                               output_data[frame_number][vehicle_id]['license_plate']['bbox_confidence'],
                                                               output_data[frame_number][vehicle_id]['license_plate']['text'],
                                                               output_data[frame_number][vehicle_id]['license_plate']['text_confidence'])
                               )
        file.close()


def is_valid_license_format(license_text):
    """
    Check if the license plate text complies with the required format.

    Args:
        license_text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(license_text) != 7:
        return False

    if (license_text[0] in string.ascii_uppercase or license_text[0] in int_to_char_mapping.keys()) and \
       (license_text[1] in string.ascii_uppercase or license_text[1] in int_to_char_mapping.keys()) and \
       (license_text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or license_text[2] in char_to_int_mapping.keys()) and \
       (license_text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or license_text[3] in char_to_int_mapping.keys()) and \
       (license_text[4] in string.ascii_uppercase or license_text[4] in int_to_char_mapping.keys()) and \
       (license_text[5] in string.ascii_uppercase or license_text[5] in int_to_char_mapping.keys()) and \
       (license_text[6] in string.ascii_uppercase or license_text[6] in int_to_char_mapping.keys()):
        return True
    else:
        return False


def format_license_text(license_text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        license_text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    formatted_license = ''
    mapping = {0: int_to_char_mapping, 1: int_to_char_mapping, 4: int_to_char_mapping, 5: int_to_char_mapping, 6: int_to_char_mapping,
               2: char_to_int_mapping, 3: char_to_int_mapping}
    for index in [0, 1, 2, 3, 4, 5, 6]:
        if license_text[index] in mapping[index].keys():
            formatted_license += mapping[index][license_text[index]]
        else:
            formatted_license += license_text[index]

    return formatted_license


def extract_license_plate(license_plate_image):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_image (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    ocr_results = ocr_reader.readtext(license_plate_image)

    for result in ocr_results:
        bbox, text, confidence = result

        text = text.upper().replace(' ', '')

        if is_valid_license_format(text):
            return format_license_text(text), confidence

    return None, None


def fetch_car(license_plate, tracked_vehicle_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        tracked_vehicle_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    found_match = False
    for index in range(len(tracked_vehicle_ids)):
        car_x1, car_y1, car_x2, car_y2, vehicle_id = tracked_vehicle_ids[index]

        if x1 > car_x1 and y1 > car_y1 and x2 < car_x2 and y2 < car_y2:
            car_index = index
            found_match = True
            break

    if found_match:
        return tracked_vehicle_ids[car_index]

    return -1, -1, -1, -1, -1