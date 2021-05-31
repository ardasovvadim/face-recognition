import math

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


class FaceMaskMaker:

    def __init__(self, args) -> None:
        if 'use_cnn' in args and args['use_cnn'] is True:
            path_to_face_detector = args["face-detector"] if "face-detector" in args else r'C:\DevEnv\Workspaces\facemask-maker\facemaskmaker\models\mmod_human_face_detector.dat'
            self.face_detector = dlib.cnn_face_detection_model_v1(path_to_face_detector)
            self.use_cnn = True
        else:
            self.face_detector = dlib.get_frontal_face_detector()
            self.use_cnn = False
        path_to_shape_predictor = args["shape-predictor"] if "shape-predictor" in args else r'C:\DevEnv\Workspaces\facemask-maker\facemaskmaker\models\shape_predictor_68_face_landmarks.dat'
        self.shape_predictor = dlib.shape_predictor(path_to_shape_predictor)

    def get_face_rectangles_by_cnn_dlib(self, image, write_labels=False):
        m_rectangles = self.face_detector(image, 1)

        if write_labels:
            image = image.copy()
            for (i, m_rect) in enumerate(m_rectangles):
                (rectangleX, rectangleY, rectangleWidth, rectangleHeight) = face_utils.rect_to_bb(m_rect.rect) if self.use_cnn else face_utils.rect_to_bb(m_rect)
                cv2.rectangle(image, (rectangleX, rectangleY), (rectangleX + rectangleWidth, rectangleY + rectangleHeight), (0, 255, 0), 2)
                cv2.putText(image, f"Face #{i + 1}", (rectangleX, rectangleY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return [m_rect.rect for m_rect in m_rectangles] if self.use_cnn else m_rectangles, image

    def get_shape68_by_dlib(self, image, rectangles, write_shape_points=False):
        if write_shape_points:
            image = image.copy()
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shapes = []

        for (i, rect) in enumerate(rectangles):
            shape = self.shape_predictor(grayImage, rect)
            shape = face_utils.shape_to_np(shape)
            shapes.append(shape)

            if write_shape_points:
                for i in range(len(shape)):
                    x, y = shape[i]
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)


        return shapes, image

    def overlay_image(self, bg_image, fg_image, x_coord, y_coord):
        result_image = bg_image.copy()
        y1, y2 = y_coord, y_coord + fg_image.shape[0]
        x1, x2 = x_coord, x_coord + fg_image.shape[1]
        fg_alpha = fg_image[:, :, 3] / 255.0
        bg_alpha = 1.0 - fg_alpha
        for c in range(0, 3):
            result_image[y1:y2, x1:x2, c] = (fg_alpha * fg_image[:, :, c] + bg_alpha * bg_image[y1:y2, x1:x2, c])

        return result_image

    def calculate_face_angel(self, shape):
        x, y = shape[8]
        a = np.asarray((x, y - 1))
        b = np.asarray(shape[8])
        c = np.asarray(shape[27])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return math.degrees(angle)

    def add_face_mask_by_shapes(self, face_image, mask_image, shapes):
        for shape in shapes:
            left_x, left_y = shape[1]
            right_x, right_y = shape[15]
            topX, top_y = shape[28]
            bottomX, bottom_y = shape[8]

            face_angel = self.calculate_face_angel(shape)
            isLeftSide = shape[27][0] < shape[8][0]
            if isLeftSide:
                face_angel *= -1

            mask_new_width, mask_new_height = right_x - left_x, bottom_y - top_y
            resized_mask_img = cv2.resize(mask_image, (mask_new_width, mask_new_height), interpolation=cv2.INTER_CUBIC)
            rotated_mask_img = imutils.rotate_bound(resized_mask_img, face_angel)

            mask_x, mask_y = (right_x - mask_new_width, right_y) if isLeftSide else (left_x - np.abs(rotated_mask_img.shape[1] - resized_mask_img.shape[1]), left_y)

            overflow_x, overflow_y = mask_x + rotated_mask_img.shape[1] - face_image.shape[1], mask_y + rotated_mask_img.shape[0] - face_image.shape[0]
            if overflow_x > 0 and overflow_y > 0:
                rotated_mask_img = cv2.resize(rotated_mask_img, (rotated_mask_img.shape[1]-overflow_x, rotated_mask_img.shape[0]-overflow_y), interpolation=cv2.INTER_CUBIC)
            elif overflow_x > 0:
                rotated_mask_img = cv2.resize(rotated_mask_img, (rotated_mask_img.shape[1]-overflow_x, rotated_mask_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            elif overflow_y > 0:
                rotated_mask_img = cv2.resize(rotated_mask_img, (rotated_mask_img.shape[1], rotated_mask_img.shape[0]-overflow_y), interpolation=cv2.INTER_CUBIC)

            face_image = self.overlay_image(face_image, rotated_mask_img, mask_x, mask_y)

        return face_image

    def add_face_mask(self, face_image, mask_image):
        rectangles, _ = self.get_face_rectangles_by_cnn_dlib(face_image)
        shapes, _ = self.get_shape68_by_dlib(face_image, rectangles)
        result_image = self.add_face_mask_by_shapes(face_image, mask_image, shapes)

        return result_image

    def add_head_line(self, face_image, shapes):
        face_image = face_image.copy()
        for shape in shapes:
            face_image = cv2.line(face_image, tuple(shape[8]), tuple(shape[27]), (255, 0, 0, 255), 2)
        return face_image


pass
