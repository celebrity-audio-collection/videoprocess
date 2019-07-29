import cv2
import numpy as np
from scipy import misc
from common import config


class CV_Tracker:

    def __init__(self, raw_img, boundary_box, series_id, lip_center, shot_count):
        if config.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if config.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if config.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if config.tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if config.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if config.tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if config.tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()

        self.bbox = (
            boundary_box[0], boundary_box[1], boundary_box[2] - boundary_box[0], boundary_box[3] - boundary_box[1])

        self.tracked = True
        self.series_name = "Series" + str(series_id)
        self.drop_count = 0
        self.valid = True
        self.start_shot = shot_count
        self.end_shot = None
        self.delta = (0, 0)
        self.sync_seq = []

        # self.raw_seq = []
        self.bbox_seq = []
        self.lip_box_seq = []

        self.last_lip_box = None
        self.update_lip_seq(raw_img, boundary_box, lip_center)
        self.tracker.init(raw_img, self.bbox)
        self.remove = False

    # save lip sequence
    def update_lip_seq(self, raw_img, boundary_box, lip_center=None):
        if lip_center is not None:
            length = int(max(boundary_box[3] - boundary_box[1], boundary_box[2] - boundary_box[0]) / 2)
            lip_box = [int(max(lip_center[1] - length, 0)), int(lip_center[1] + length),
                       int(max(lip_center[0] - length, 0)), int(lip_center[0] + length)]
            lip_center_picture = raw_img[int(lip_box[0]):int(lip_box[1]), int(lip_box[2]):int(lip_box[3]), :]
            lip_center_picture = misc.imresize(lip_center_picture, (224, 224), interp='bilinear')
            self.sync_seq.append(lip_center_picture)

            if config.debug:
                self.bbox_seq.append((boundary_box[0], boundary_box[1], boundary_box[2], boundary_box[3]))
                self.lip_box_seq.append(lip_box)

            self.last_lip_box = lip_box
        else:
            lip_box = self.last_lip_box
            lip_box = [int(max(lip_box[0] + self.delta[1], 0)), int(max(lip_box[1] + self.delta[0], 0)),
                       int(max(lip_box[2] + self.delta[1], 0)), int(max(lip_box[3] + self.delta[0], 0))]
            lip_center_picture = raw_img[lip_box[0]:lip_box[1], lip_box[2]:lip_box[3], :]
            lip_center_picture = misc.imresize(lip_center_picture, (224, 224), interp='bilinear')
            self.sync_seq.append(lip_center_picture)

            if config.debug:
                self.bbox_seq.append((lip_box[0], lip_box[1], lip_box[2], lip_box[3]))
                self.lip_box_seq.append(lip_box)


            self.last_lip_box = lip_box

    def update(self, raw_img, shot_count):
        self.tracked, new_bbox = self.tracker.update(raw_img)
        if self.tracked is True:
            self.delta = (new_bbox[0] - self.bbox[0] + (new_bbox[2] - self.bbox[2]) / 2,
                          new_bbox[1] - self.bbox[1] + (new_bbox[3] - self.bbox[3]) / 2)
            self.bbox = new_bbox
            self.valid = False
        else:
            self.end_shot = shot_count
            self.remove = True
        return self.tracked, self.bbox

    # judge if tracker box overlaps with face detection box
    def is_tracking(self, center):
        tracking_area = self.bbox
        area_center = (tracking_area[0] + tracking_area[2] / 2, tracking_area[1] + tracking_area[3] / 2)
        distance = np.sqrt(np.sum(np.square(np.subtract(list(area_center), list(center)))))
        # print("distance", distance)
        if distance < 0.3 * np.sqrt(np.sum(np.square([tracking_area[2], tracking_area[3]]))):
            return True
        else:
            return False

    def is_valid(self, center):
        if self.valid is True:
            return True
        tracking_area = self.bbox
        area_center = (tracking_area[0] + tracking_area[2] / 2, tracking_area[1] + tracking_area[3] / 2)
        distance = np.sqrt(np.sum(np.square(np.subtract([center[1], center[0]], list(area_center)))))
        if distance < 0.3 * np.sqrt(np.sum(np.square([tracking_area[2], tracking_area[3]]))):
            self.valid = True
            self.drop_count = 0
            return True
        else:
            return False

    def get_lip_seq(self):
        return self.sync_seq

    def get_drop_count(self):
        return self.drop_count

    def get_shot_range(self):
        return self.start_shot, self.end_shot

    def drop(self):
        if self.drop_count >= config.patience:
            return True
        else:
            return False

    def set_end_shot(self, shot_count):
        self.end_shot = shot_count
