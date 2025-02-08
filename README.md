# Vehicle Detection and License Plate Recognition

This project demonstrates vehicle detection and license plate recognition using YOLOv8 and a license plate detection model. It includes a video processing pipeline for vehicle tracking and license plate detection.

## Model

- **YOLOv8n**: A pre-trained YOLOv8 model was used to detect vehicles.
- **License Plate Detector**: A separate model was used to detect license plates. The model was trained using YOLOv8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).


## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
python -m venv venv
```
* Activate the environment
``` bash
venv\Scripts\activate
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run main.py with the sample video file to generate the test.csv file 
``` python
python main.py
```
* Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python process_data.py
```

* Finally run the visualize.py passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
```python
python visualize.py
```
## License

**Copyright © 2025 DevOptima (Mustafa Dadikhi)**
All rights reserved. Unauthorized use, reproduction, or distribution of this material is prohibited without prior written permission from the copyright holder.
