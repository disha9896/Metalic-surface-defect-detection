from defect_detection import ObjectDetection
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

od = ObjectDetection()
od.load_model("model/best.pt")

# object classes
classNames = ["crease", "crescent_gap", "d" ,"inclusion", "oil_spot","punching_hole",
               "rolled_pit", "silk_spot","waist folding","water_spot","welding_line"]

cam = cv2.VideoCapture(r"video_frames/2024-04-09_10-36-32.mp4") 
frameTime = 20
success, img = cam.read()
height, width, layers = img.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(r"Results\videos\output2.avi", fourcc, 20.0, (width, height))

while(cam.isOpened()):
    success, img = cam.read()
    results = od.model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

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
    video.write(img)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
video.release()