"""
Yolov9 model training code
"""

from defect_detection import ObjectDetection
import os

data_dir = os.path.join(os.path.dirname(__file__), "Dataset")
od = ObjectDetection()

try:
    # load the dataset 
    od.load_dataset(data_dir)
    #load the pre-trained model
    od.load_model("model/best.pt")
    #start training
    od.train()
    
except ValueError as e:
    print(f"Dataset format validation error: {e}")