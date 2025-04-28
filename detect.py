from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0) # this is for using camera. For testing purposes, the '0' can be replaced with the link of test video

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)

TARGET_CLASSES = [0, 2, 9, 10, 11]  # person, car, traffic light, fire hydrant, stop sign

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in TARGET_CLASSES:
                frame = result.plot()

    frame_resized = cv2.resize(frame, (1024, 768))

    cv2.imshow("YOLOv8 Object Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
