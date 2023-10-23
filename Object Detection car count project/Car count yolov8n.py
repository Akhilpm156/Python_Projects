import math
from ultralytics import YOLO
import cv2
import cvzone
from sort import *


model = YOLO('yolov8n.pt')

# All class Name available

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
visual = cv2.VideoCapture("cars.mp4")
mask1 = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
line_a = [400, 297, 673, 297]
total_count1 = []

while True:
    Success, img = visual.read()
    new_view = cv2.bitwise_and(img, mask1)
    result = model(new_view, stream=True)
    Graphic_images = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, Graphic_images, (0, 0))

    detections = np.empty((0, 5))

    for i in result:
        boxes = i.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # confidence
            confidence = (math.ceil(box.conf[0] * 100) / 100)

            # Current Class_Name
            cls = int(box.cls[0])
            current_class = classNames[cls]
            if current_class == 'car' or current_class == 'truck' or current_class == 'bus' or\
                    current_class == 'motorbike' and confidence > 0.3:
                # cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0, x1), max(30, y1-5)),
                #                    scale=0.70, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2, colorR=(0, 0, 255))

                current_array = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_array))

    result_tracker = tracker.update(detections)

    cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (0, 0, 255), 5)

    for res in result_tracker:
        x1, y1, x2, y2, id1 = res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(res)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=2, colorR=(0, 0, 255))

        cvzone.putTextRect(img, f'{int(id1)}', (max(0, x1), max(30, y1 - 5)),
                           scale=1, thickness=1, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)

        if line_a[0] < cx < line_a[2] and line_a[1] - 15 < cy < line_a[1] + 15:
            if total_count1.count(id1) == 0:
                total_count1.append(id1)
                cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(total_count1)}', (50, 50))
    cv2.putText(img, str(len(total_count1)), (250, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (50, 50, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
