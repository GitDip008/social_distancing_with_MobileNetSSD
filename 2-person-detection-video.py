import cv2
import datetime
import imutils
import numpy as np

proto_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

network = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def frame_count():
    cap = cv2.VideoCapture("test_video.mp4")

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        suc, img = cap.read()
        img = imutils.resize(img, width=400)


        # The detection part
        # print(img.shape)        # (450, 800, 3)
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)
        network.setInput(blob)
        detections = network.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                bounding_box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = bounding_box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Counting the FPS
        total_frames += 1
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time

        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = total_frames / time_diff.seconds

        fps_text = "FPS: {:.2F}".format(fps)
        cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
frame_count()