"""
Predict multiple images. Pass folder of images and get results with detected defects on it. 
Give a specific path to store the results otherwise it will be stored in the pre determined path.
"""

from defect_detection import ObjectDetection
import os

od = ObjectDetection()

data_dir = os.path.join(os.path.dirname(__file__), "Dataset", "test", "images")
od.predict_multiple("model/best.pt", data_dir)