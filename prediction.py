# prediction on a specific image
from defect_detection import ObjectDetection
import os

od = ObjectDetection()

od.model_prediction("model/best.pt", "Dataset/test/images/img_01_4402724300_00001_jpg.rf.7bcfacc21bccbec82f4e03e748484b35.jpg")


