from defect_detection import ObjectDetection
import os

od = ObjectDetection()

data_dir = os.path.join(os.path.dirname(__file__), "Dataset", "test", "images")
od.predict_multiple("model/best.pt", data_dir)