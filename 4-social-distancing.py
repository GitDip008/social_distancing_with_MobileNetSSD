import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from itertools import combinations
import math


proto_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

network = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)
# network.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(50, 90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def frame_count():
    cap = cv2.VideoCapture("pedestrians.mp4")

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    centroid_dict = dict()

    while True:
        suc, img = cap.read()
        img = imutils.resize(img, width=600)


        # The detection part
        # print(img.shape)        # (450, 800, 3)
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (W, H), 127.5)
        network.setInput(blob)
        detections = network.forward()
        rects_coordinates = []

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                bounding_box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = bounding_box.astype("int")
                rects_coordinates.append(bounding_box)

        bounding_boxes = np.array(rects_coordinates)
        bounding_boxes = bounding_boxes.astype(int)
        rects = non_max_suppression_fast(bounding_boxes, 0.3)

        objects = tracker.update(rects)     # objects contain the object id and the bounding box

        for (object_id, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int( (x1 + x2) / 2.0)
            cY = int( (y1 + y2) / 2.0)

            centroid_dict[object_id] = (cX, cY, x1, y1, x2, y2)

            text = "ID: {}".format(object_id)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        red_zone = []
        for (id1, pt1), (id2, pt2) in combinations(centroid_dict.items(), 2):
            dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone:
                    red_zone.append(id1)
                if id2 not in red_zone:
                    red_zone.append(id2)

        for id, box in centroid_dict.items():
            if id in red_zone:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

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