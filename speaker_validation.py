from common import config
from syncnet import SyncNetInstance
import numpy as np


class SpeakerValidation:

    def __init__(self):
        self.model = SyncNetInstance.SyncNetInstance()
        self.model.loadParameters(config.syncnet_model)

    def form_convert(self, frame_id):
        h = int(frame_id / 90000)
        rest = frame_id % 90000
        m = int(rest / 1500)
        rest = rest % 1500
        s = int(rest / 25)
        lf = rest % 25
        return "{:d}:{:d}:{:d}:{:d}".format(h, m, s, lf)

    def evaluate(self, video_fps, image_seq, audio_seq):
        if len(image_seq) <= 6:
            return None, np.array([0]), None
        offset, confidence, dists_npy = self.model.evaluate_part(video_fps, image_seq, audio_seq)
        return offset, confidence, dists_npy

    def verification(self, confidence, start_shot, logfile):
        candidates = []
        processed = -1
        for index in range(0, confidence.shape[0]):
            if index <= processed:
                continue
            if confidence[index] >= config.starting_confidence:
                slice_start = start_shot + index
                while index < confidence.shape[0] and confidence[index] >= config.patient_confidence:
                    index += 1
                processed = index
                slice_end = start_shot + index + 6
                candidates.append((slice_start, slice_end))
                logfile.writelines([self.form_convert(slice_start) + "\t" + self.form_convert(slice_end) + "\n"])
        return candidates
