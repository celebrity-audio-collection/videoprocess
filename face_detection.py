import numpy as np
from common import config
import platform
if platform.platform().find('Windows') >= 0:
    from RetinaFaceModel.insightface.RetinaFace.retinaface import RetinaFace
elif platform.platform().find('Linux') >= 0:
    from RetinaFaceModel.insightface.RetinaFace_linux.retinaface import RetinaFace


class FaceDetection:

    def __init__(self):
        print('Loading face detection model, this may take a second...')
        self.retinaface_detector = RetinaFace(config.retinaface_model, 0, config.gpuid, 'net3')
        self.im_scale_updated = False
        self.im_scale = None

    def update_im_scale(self, raw_img):
        scales = config.detect_scale
        im_shape = raw_img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        # im_scale = 1.0
        # if im_size_min>target_size or im_size_max>max_size:
        self.im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(self.im_scale * im_size_max) > max_size:
            self.im_scale = float(max_size) / float(im_size_max)
        self.im_scale_updated = True

    def update(self, raw_img):
        # using retina face
        if not self.im_scale_updated:
            self.update_im_scale(raw_img)
        bboxes, landmarks = self.retinaface_detector.detect(raw_img, config.thresh, scales=[self.im_scale],
                                                            do_flip=False)
        return bboxes, landmarks
