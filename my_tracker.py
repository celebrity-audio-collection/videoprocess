import cv2
import numpy as np
from scipy import misc
from common import config


class MyTracker:

    def __init__(self, raw_img, boundarybox, serieid, lipcenter, shotcount):
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

        self.tracker = cv2.TrackerMOSSE_create()
        self.bbox = (boundarybox[0], boundarybox[1], boundarybox[2] - boundarybox[0], boundarybox[3] - boundarybox[1])

        self.tracked = True
        self.serie_name = "series" + str(serieid)
        self.drop_count = 0
        self.valid = True
        self.startshot = shotcount
        self.endshot = None
        self.delta = (0, 0)
        # self.seqlist = []
        self.syncseq = []
        self.lastlipbox = None
        self.update_lip_seq(raw_img, boundarybox, lipcenter)
        self.tracker.init(raw_img, self.bbox)
        self.remove = False

    def update_lip_seq(self, raw_img, boundarybox, lipcenter=None):
        if lipcenter is not None:
            length = int(max(boundarybox[3] - boundarybox[1], boundarybox[2] - boundarybox[0]) / 2)
            lipbox = [max(lipcenter[1] - length, 0), lipcenter[1] + length,
                      max(lipcenter[0] - length, 0), lipcenter[0] + length]
            lipcenter_picture = raw_img[int(lipbox[0]):int(lipbox[1]), int(lipbox[2]):int(lipbox[3]), :]
            lipcenter_picture = misc.imresize(lipcenter_picture, (224, 224), interp='bilinear')
            self.syncseq.append(lipcenter_picture)
            self.lastlipbox = lipbox
        else:
            lipbox = self.lastlipbox
            lipbox = [int(max(lipbox[0] + self.delta[1], 0)), int(max(lipbox[1] + self.delta[0], 0)),
                      int(max(lipbox[2] + self.delta[1], 0)), int(max(lipbox[3] + self.delta[0], 0))]
            lipcenter_picture = raw_img[lipbox[0]:lipbox[1], lipbox[2]:lipbox[3], :]
            lipcenter_picture = misc.imresize(lipcenter_picture, (224, 224), interp='bilinear')
            self.syncseq.append(lipcenter_picture)
            self.lastlipbox = lipbox

    def update(self, raw_img, shotcount):
        self.tracked, newbbox = self.tracker.update(raw_img)
        if self.tracked is True:
            self.delta = (newbbox[0] - self.bbox[0] + (newbbox[2] - self.bbox[2]) / 2,
                          newbbox[1] - self.bbox[1] + (newbbox[3] - self.bbox[3]) / 2)
            self.bbox = newbbox
            self.valid = False
        else:
            self.endshot = shotcount
            self.remove = True
        return self.tracked, self.bbox

    def is_tracking(self, center):
        tracking_area = self.bbox
        # print('boxcenter',boxcenter)
        areacenter = (tracking_area[0] + tracking_area[2] / 2, tracking_area[1] + tracking_area[3] / 2)
        # print('areacenter', areacenter)
        distance = np.sqrt(np.sum(np.square(np.subtract(list(areacenter), list(center)))))
        # print("distance",distance)
        if distance < 0.3 * np.sqrt(np.sum(np.square([tracking_area[2], tracking_area[3]]))):
            return True
        else:
            return False

    def is_valid(self, center):
        if self.valid is True:
            return True
        tracking_area = tracking_area = self.bbox
        # print('boxcenter',boxcenter)
        areacenter = (tracking_area[0] + tracking_area[2] / 2, tracking_area[1] + tracking_area[3] / 2)
        distance = np.sqrt(np.sum(np.square(np.subtract([center[1], center[0]], list(areacenter)))))
        # print("distance",distance)
        # if distance < 0.3 * np.sqrt(np.sum(np.square([b[2] - b[0], b[3] - b[1]]))):
        if distance < 0.3 * np.sqrt(np.sum(np.square([tracking_area[2], tracking_area[3]]))):
            self.valid = True
            self.drop_count = 0
            return True
        else:
            return False

    def get_lip_seq(self):
        return self.syncseq

    def get_drop_count(self):
        return self.drop_count

    def get_shot_range(self):
        return (self.startshot, self.endshot)

    def drop(self):
        if self.drop_count >= 8:
            return True
        else:
            return False

    def set_endshot(self, shotcount):
        self.endshot = shotcount
