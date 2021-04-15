import cv2
import datetime
import imutils

def frame_count():
    cap = cv2.VideoCapture("test_video.mp4")

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        suc, img = cap.read()
        img = imutils.resize(img, width=800)
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