import cv2
import numpy as np
import imutils

proto_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

network = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def detect():
    img = cv2.imread("people.jpg")
    img = imutils.resize(img, width=600)

    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)

    network.setInput(blob)

    detections = network.forward()
    # print(detections.shape)

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            bounding_box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, starty, endX, endY) = bounding_box.astype("int")

            cv2.rectangle(img, (startX, starty), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect()