import argparse
import time
import cv2
import imutils
from imutils.video import VideoStream

import mask_detector

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default=r"C:\DevEnv\Workspaces\facemask-maker\mask_detector\models\mask_detector_2.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face mask detector model...")

detector = mask_detector.MaskDetectorModel({'model_path': args['model'], 'ssn_confidence': args['confidence']})

print("[INFO] starting video stream...")

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640, height=480)

    _, frame = detector.detect_face_mask(frame, True)

    cv2.imshow("Result", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
