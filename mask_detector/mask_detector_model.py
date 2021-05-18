import datetime
import time

import cv2.dnn
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
SHAPE = (224, 224, 3)
ACC_THRESHOLD = 0.99

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > ACC_THRESHOLD:
            print(f"\nReached %{ACC_THRESHOLD * 100} accuracy, so stopping training")
            self.model.stop_training = True


class MaskDetectorModel:

    def __init__(self, args={}) -> None:
        self.model = None
        self.H = None
        self.train_data_generator = None
        self.test_data_generator = None
        self.STEP_SIZE_TRAIN = None
        self.STEP_SIZE_VALID = None

        ssn_dep_path = args['ssn_dep_path'] if 'ssn_dep_path' in args else r'C:\DevEnv\Workspaces\facemask-maker\mask_detector\models\face_detector\deploy.prototxt'
        ssn_weights_path = args['ssn_weights_path'] if 'ssn_weights_path' in args else r'C:\DevEnv\Workspaces\facemask-maker\mask_detector\models\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
        self.ssn_face_detector = cv2.dnn.readNet(ssn_dep_path, ssn_weights_path)
        self.ssn_confidence = args['ssn_confidence'] if 'ssn_confidence' in args else 0.5

        super().__init__()

    def train(self, dataset_path, with_stop_callback=True):
        print('[INFO] creating train and test dataset generators...')

        self.__get_train_and_test_data_gens(dataset_path)

        print('[INFO] operation was completed successfully')

        self.STEP_SIZE_TRAIN = self.train_data_generator.n // self.train_data_generator.batch_size
        self.STEP_SIZE_VALID = self.test_data_generator.n // self.test_data_generator.batch_size

        print('[INFO] compiling model...')

        self.__generate_and_compile_model()

        print('[INFO] model was compiled successfully')

        print('[INFO] training...')
        start_time = time.time()

        callbacks = MyCallback() if with_stop_callback else []

        self.H = self.model.fit(
            self.train_data_generator,
            steps_per_epoch=self.STEP_SIZE_TRAIN,
            validation_data=self.test_data_generator,
            validation_steps=self.STEP_SIZE_VALID,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        end_time = time.time()
        print('[INFO] training was completed successfully')
        print(f'[INFO] training time: {str(datetime.timedelta(seconds=end_time - start_time))}')

        return

    def save_model(self, model_save_path):
        if self.model:
            self.model.save(model_save_path, save_format="h5")

    def __get_train_and_test_data_gens(self, dataset_path):
        image_data_generator = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )

        self.train_data_generator = image_data_generator.flow_from_directory(
            dataset_path,
            target_size=SHAPE[:2],
            batch_size=BS,
            class_mode='categorical',
            subset='training'
        )

        self.test_data_generator = image_data_generator.flow_from_directory(
            dataset_path,
            target_size=SHAPE[:2],
            batch_size=BS,
            class_mode='categorical',
            subset='validation'
        )

        return

    def __generate_and_compile_model(self):
        baseModel = MobileNetV2(weights="imagenet",
                                include_top=False,
                                input_tensor=Input(shape=SHAPE))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        self.model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return

    def print_classification_report(self):
        predictions = self.model.predict_generator(self.test_data_generator, steps=self.STEP_SIZE_VALID, verbose=1)
        y_predictions = np.argmax(predictions, axis=-1)
        print(classification_report(self.test_data_generator.classes[:len(y_predictions)], y_predictions, target_names=self.test_data_generator.class_indices))
        return y_predictions

    def load_from(self, model_path):
        self.model = load_model(model_path)

    def predict(self, data):
        if not self.model:
            return []
        return self.model.predict(data, verbose=1)

    def detect_face(self, img, write_labels=False):
        if not self.ssn_face_detector:
            print("SSN wasn't initialized")

        face_rectangles = []
        img = img.copy()
        color = (0, 255, 0)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.ssn_face_detector.setInput(blob)
        detections = self.ssn_face_detector.forward()
        face_i = 0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= self.ssn_confidence:
                face_i += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face_rectangles.append((startX, startY, endX, endY))

                if write_labels:
                    img = cv2.putText(img, f'Face {face_i}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    img = cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

        return face_rectangles, img