import cv2
from facemaskmaker.facemaskmaker import FaceMaskMaker

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    args = {
        "showBorder": True,
        "addMask": True,
        "addHeadLine": False,
        "showPoints": True,
        "predictorPath": "predictors/shape_predictor_68_face_landmarks.dat",
        "maskImagePath": "imgs/masks-examples/mask-1.png"
    }
    faceMaskMaker = FaceMaskMaker(args)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frameImage = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        faceMaskMaker.process(frameImage)

        # Display the resulting frame
        cv2.imshow('frame', frameImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()