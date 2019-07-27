# import keras
import numpy as np
# import keras_retinanet.bin  # noqa: F401
# import keras_retinanet.models
from common import config
from RetinaFaceModel.insightface.RetinaFace.retinaface import RetinaFace

__package__ = "keras_retinanet.bin"


class FaceDetection:

    def __init__(self, modelpath=config.face_detection_model):
        print('Loading face detection model, this may take a second...')
        # self.model = keras_retinanet.models.load_model(modelpath)
        self.retinaface_detector = RetinaFace('model/retinaface_model/mnet.25/mnet.25', 0, config.gpuid, 'net3')

    def update(self, raw_img):
        # use retina face
        scales = [360, 640]
        im_shape = raw_img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        # im_scale = 1.0
        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        bboxes, landmarks = self.retinaface_detector.detect(raw_img, config.thresh, scales=[im_scale], do_flip=False)
        return bboxes, landmarks

        ## use retina net
        # if keras.backend.image_data_format() == 'channels_first':
        #     raw_img = raw_img.transpose((2, 0, 1))
        #
        # boxes, scores, labels = self.model.predict(np.expand_dims(raw_img, axis=0))[:3]
        #
        # # correct boxes for image scale
        # # boxes /= scale
        #
        # # select indices which have a score above the threshold
        # indices = np.where(scores[0, :] > 0.05)[0]
        #
        # # select those scores
        # scores = scores[0][indices]
        #
        # # find the order with which to sort the scores
        # scores_sort = np.argsort(-scores)[:100]
        #
        # # select detections
        # image_boxes = boxes[0, indices[scores_sort], :]
        #
        # image_scores = scores[scores_sort]
        # image_labels = labels[0, indices[scores_sort]]
        #
        # image_detections = np.concatenate(
        #     [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        #
        # selection = np.where(image_scores > 0.5)[0]
        #
        # return [ np.array(image_boxes[i, :].astype(int)) for i in selection]
