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

        self.tracked = True                                     # 在某次追踪中，是否追踪到目标
        self.series_name = "Series" + str(series_id)            # tracker 名称
        self.drop_count = 0                                     # 记录到当前帧为止，追踪区域中没有POI的帧数，区域中出现POI后清零
        self.valid = True                                       # 辅助判断当前帧追踪区域中是否存在POI
        self.start_shot = shot_count                            # 追踪开始帧
        self.end_shot = None                                    # 追踪结束帧
        self.delta = (0, 0)                                     # 追踪区域偏移量，用于在追踪区域中没有POI的情况下移动嘴唇节选框，裁剪图片
        self.sync_seq = []                                      # tracker 追踪过程中，获得的以嘴唇为中心的正方形图片序列，用于syncnet输入

        self.bbox_seq = []                                      # debug 模式下信息记录
        self.lip_box_seq = []                                   #

        self.last_lip_box = None                                # 前一帧中的嘴唇区域节选框
        self.update_lip_seq(raw_img, boundary_box, lip_center)
        self.tracker.init(raw_img, self.bbox)                   # 初始化tracker
        self.remove = False

    # save lip sequence
    '''
        @requires raw_img ！= None
        @modifies self.sync_seq
        @effects  如果lip_center不为空，利用人脸检测的长边以及lip_center切割嘴唇图片
                  否则利用上一次的切分边界以及追踪偏移切割嘴唇图片，若偏移超出原始图片范围，将图片置零
    '''
    def update_lip_seq(self, raw_img, boundary_box, lip_center=None):
        if lip_center is not None:
            length = int(max(boundary_box[3] - boundary_box[1], boundary_box[2] - boundary_box[0]) / 2)
            length *= 1.2
            lip_box = [int(max(lip_center[1] - length, 0)), int(lip_center[1] + length),
                       int(max(lip_center[0] - length, 0)), int(lip_center[0] + length)]
            lip_center_picture_raw = raw_img[int(lip_box[0]):int(lip_box[1]), int(lip_box[2]):int(lip_box[3]), :]
            try:
                lip_center_picture = cv2.resize(lip_center_picture_raw, (224, 224))
            except Exception:
                print("Tracker: misc resize error")
                print(lip_box)
                print(length)
                print(lip_center_picture_raw)
                exit(-1)
            if config.debug:
                cv2.imshow('lip', lip_center_picture)
                cv2.waitKey(40)
            self.sync_seq.append(lip_center_picture)
            if config.debug:
                self.bbox_seq.append((boundary_box[0], boundary_box[1], boundary_box[2], boundary_box[3]))
                self.lip_box_seq.append(lip_box)

            self.last_lip_box = lip_box
        else:
            lip_box = self.last_lip_box
            lip_box = [int(max(lip_box[0] + self.delta[1], 0)), int(max(lip_box[1] + self.delta[1], 0)),
                       int(max(lip_box[2] + self.delta[0], 0)), int(max(lip_box[3] + self.delta[0], 0))]
            lip_center_picture_raw = raw_img[lip_box[0]:lip_box[1], lip_box[2]:lip_box[3], :]
            if len(lip_center_picture_raw) == 0:
                lip_center_picture_raw = np.zeros((224, 224, 3))
            try:
                lip_center_picture = cv2.resize(lip_center_picture_raw, (224, 224))
            except Exception:
                print("Tracker: misc resize error")
                print(lip_box)
                print(self.delta)
                print(lip_center_picture_raw)
                exit(-1)
            if config.debug:
                cv2.imshow('lip', lip_center_picture)
                cv2.waitKey(40)
            self.sync_seq.append(lip_center_picture)

            if config.debug:
                self.bbox_seq.append((lip_box[0], lip_box[1], lip_box[2], lip_box[3]))
                self.lip_box_seq.append(lip_box)

            self.last_lip_box = lip_box

    # track
    '''
        @requires raw_img!= None, shot_count >= 0
        @modifies self.bbox, self.valid, self.remove, self.tracked, self.end_shot
        @effects  追踪一帧图片，如果追踪到，更新追踪偏移，否则将追踪器标记为待移除
    '''
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

    # 判断该人脸是否已经被tracker追踪
    def is_tracking(self, center):
        tracking_area = self.bbox
        area_center = (tracking_area[0] + tracking_area[2] / 2, tracking_area[1] + tracking_area[3] / 2)
        distance = np.sqrt(np.sum(np.square(np.subtract(list(area_center), list(center)))))
        # print("distance", distance)
        if distance < 0.3 * np.sqrt(np.sum(np.square([tracking_area[2], tracking_area[3]]))):
            return True
        else:
            return False

    # 判断该tracker 是否追踪到了某个人脸候选框
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
