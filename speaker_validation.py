from common import config
from syncnet import SyncNetInstance
import numpy as np
import cv2
class FpeakerValidation:

    def __init__(self):
        self.model = SyncNetInstance.SyncNetInstance()
        self.model.loadParameters(config.syncnet_model)

    def evaluate(self, video_fps, imageseq, audioseq):
        if len(imageseq) <= 6:
            return None, np.array([0]), None
        offset, confidence, dists_npy =  self.model.evaluate_part(video_fps,imageseq,audioseq)
        # if len(imageseq) != len (confidence):
        #     print("WOW")
        #     print(len(imageseq),len (confidence))
        #     exit(10)
        return offset, confidence, dists_npy

    def verification(self, confidence, startshot, logfile):
        # print(confidence)
        canditates = []
        processed = -1
        for index in range(0, confidence.shape[0]):
            if index <= processed:
                continue
            if confidence[index] >= config.starting_confidence:
                stshot = startshot + index
                while (index < confidence.shape[0] and confidence[index] >= config.patient_confidence):
                    index += 1
                processed = index
                edshot = startshot + index + 6
                canditates.append((stshot, edshot))
                logfile.writelines([str(stshot) + ":" + str(edshot) + "\n"])
        return canditates
