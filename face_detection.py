import keras
import numpy as np
import keras_retinanet.bin  # noqa: F401
import keras_retinanet.models
from common import config
__package__ = "keras_retinanet.bin"

class FaceDetection:


    def __init__(self, modelpath = config.face_detection_model):
        print('Loading face detection model, this may take a second...')
        self.model = keras_retinanet.models.load_model(modelpath)

    def update(self, raw_img):
        if keras.backend.image_data_format() == 'channels_first':
            raw_img = raw_img.transpose((2, 0, 1))

        boxes, scores, labels = self.model.predict(np.expand_dims(raw_img, axis=0))[:3]

        # correct boxes for image scale
        # boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > 0.05)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:100]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]

        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]

        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        selection = np.where(image_scores > 0.5)[0]

        return [ np.array(image_boxes[i, :].astype(int)) for i in selection]



