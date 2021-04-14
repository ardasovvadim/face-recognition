import cv2
import imutils

# def addImageOnImage(img1, img2, coords)

if __name__ == '__main__':
    print('weighted-images')
    img1 = cv2.imread('imgs/example-1.jpg')
    img2 = cv2.imread('imgs/masks-examples/mask-1.png')
    rows, cols, channels = img2.shape
    img1[0:rows, 0:cols] = img2

    # originalImg = cv2.imread('imgs/masks-examples/mask-1.png')
    # print(originalImg.shape)
    # resizedImg = imutils.resize(originalImg, width=100)
    # print(resizedImg.shape)
    # cv2.imshow('result', resizedImg)
    # cv2.imshow('origin', originalImg)
    cv2.imshow('resutl', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()