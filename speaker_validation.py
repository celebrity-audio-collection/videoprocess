from common import config
from syncnet import SyncNetInstance
import numpy as np


class SpeakerValidation:

    def __init__(self):
        self.model = SyncNetInstance.SyncNetInstance()
        self.model.loadParameters(config.syncnet_model)

    def evaluate(self, video_fps, imageseq, audioseq):
        if len(imageseq) <= 6:
            return None, np.array([0]), None
        offset, confidence, dists_npy = self.model.evaluate_part(video_fps, imageseq, audioseq)
        # if len(imageseq) != len (confidence):
        #     print(len(imageseq),len (confidence))
        #     exit(10)
        return offset, confidence, dists_npy

    def verification(self, confidence, start_shot, logfile):
        # print(confidence)
        candidates = []
        processed = -1
        for index in range(0, confidence.shape[0]):
            if index <= processed:
                continue
            if confidence[index] >= config.starting_confidence:
                start_shot = start_shot + index
                while index < confidence.shape[0] and confidence[index] >= config.patient_confidence:
                    index += 1
                processed = index
                end_shot = start_shot + index + 6
                candidates.append((start_shot, end_shot))
                logfile.writelines([str(start_shot) + ":" + str(end_shot) + "\n"])
        return candidates
