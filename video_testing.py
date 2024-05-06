"""
mimicking real time video defect detection process by adding a video and storeing the defects detected video.
"""

from defect_detection import ObjectDetection
import cv2
import math 
import os

od = ObjectDetection()
od.load_model("model/best.pt")

video_path = r"video_frames/2024-04-09_10-24-26.mp4"
output_path = r"results/videos/output2.avi"

# object classes
classNames = ["crease", "crescent_gap", "d" ,"inclusion", "oil_spot","punching_hole",
               "rolled_pit", "silk_spot","waist folding","water_spot","welding_line"]

cam = cv2.VideoCapture(video_path)

frameTime = 20
success, img = cam.read()
height, width, layers = img.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


while(cam.isOpened()):
    success, img = cam.read()
    results = od.model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print(confidence)
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
    video_writer.write(img)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_writer.release()
