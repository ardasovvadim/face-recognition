import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


class FaceMaskMaker:

    def __init__(self, args) -> None:
        self.showPoints = args["showPoints"]
        self.showBorder = args["showBorder"]
        self.addMask = args["addMask"]
        self.addHeadLine = args["addHeadLine"]

        self.maskImg = cv2.imread(args["maskImagePath"])
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(args["predictorPath"])

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            if self.showBorder:
                (shapeX, shapeY, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (shapeX, shapeY), (shapeX + w, shapeY + h), (0, 255, 0), 2)
                cv2.putText(image, "Face #{}".format(i + 1), (shapeX - 10, shapeY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            if self.addMask:
                self.addMaskToImage(image, shape)

            if self.showPoints:
                for i in range(len(shape)):
                    x, y = shape[i]
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            if self.addHeadLine:
                x28, y28 = shape[27]
                x9, y9 = shape[8]
                cv2.line(image, (x28, y28), (x9, y9), (0, 0, 255), 2)

        return image

        pass

    def addMaskToImage(self, image, shape):
        # get size of mask
        x2, y2 = shape[0]
        x16, y16 = shape[15]
        x9, y9 = shape[8]
        maskWidth = x16 - x2
        maskHeight = y9 - y2
        x29, y29 = shape[28]
        x28, y28 = shape[27]

        persY = 0.08219178082191780821917808219178
        persX = 0.08974358974358974358974358974359
        deltaY = int(maskHeight * persY)
        deltaX = int(maskWidth * persX)
        maskWidth += deltaX * 2
        resizedMaskImg = cv2.resize(self.maskImg, (maskWidth, maskHeight), interpolation=cv2.INTER_CUBIC)
        angel = np.arctan2(y28 - y9, x28 - x9) * 2
        maskHeight, maskWidth, maskChannel = resizedMaskImg.shape
        resizedMaskImg = imutils.rotate(resizedMaskImg, angel, (maskWidth / 2, maskHeight))
        maskHeight, maskWidth, maskChannel = resizedMaskImg.shape
        roi = image[y29 - deltaY:y29 + maskHeight - deltaY, x2 - deltaX:x2 + maskWidth - deltaX]
        maskGrey = cv2.cvtColor(resizedMaskImg, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(maskGrey, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        imgBg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        imgFg = cv2.bitwise_and(resizedMaskImg, resizedMaskImg, mask=mask)
        resultMask = cv2.add(imgBg, imgFg)
        image[y29 - deltaY:y29 + maskHeight - deltaY, x2 - deltaX:x2 + maskWidth - deltaX] = resultMask

        pass