import argparse
import cv2
import mask_detector

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default=r"C:\DevEnv\Workspaces\facemask-maker\mask_detector\models\mask_detector_2.model", help="Path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence of model")
ap.add_argument("-i", "--image", required=True, type=str, help="Path to image file")
ap.add_argument("-r", "--result", type=str, help="Path to result image file")
args = vars(ap.parse_args())

print("[INFO] loading face mask detector model...")

detector = mask_detector.MaskDetectorModel({'model_path': args['model'], 'ssn_confidence': args['confidence']})

print("[INFO] starting video stream...")

image = cv2.imread(args['image'])
_, image = detector.detect_face_mask(image, True)

cv2.imshow("Result", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

if 'result' in args:
    cv2.imwrite(args['result'], image)
