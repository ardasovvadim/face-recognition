import cv2
import imutils

from facemaskmaker import FaceMaskMaker

args = {
        "showBorder": False,
        "addMask": True,
        "addHeadLine": False,
        "showPoints": True,
        "predictorPath": "predictors/shape_predictor_68_face_landmarks.dat",
        "maskImagePath": "imgs/masks-examples/mask-1.png"
    }
model = FaceMaskMaker(args)
image = cv2.imread('imgs/example-2.jpg')
# image = imutils.resize(image, width=400)
outImage = model.process(image)

cv2.imwrite('result-ex-3-points.jpg', image)
cv2.imshow("Output", outImage)
cv2.waitKey(0)
