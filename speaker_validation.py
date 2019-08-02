from common import config
from syncnet import SyncNetInstance
import numpy as np

'''
    说话人检测，采用 syncnet 进行唇动与语音对齐
'''
class SpeakerValidation:

    def __init__(self):
        self.model = SyncNetInstance.SyncNetInstance()
        self.model.loadParameters(config.syncnet_model)

    '''
        输出格式装换 帧号 -> 时:分:秒:帧
    '''
    def form_convert(self, frame_id):
        h = int(frame_id / 90000)
        rest = frame_id % 90000
        m = int(rest / 1500)
        rest = rest % 1500
        s = int(rest / 25)
        lf = rest % 25
        return "{:0>2d}:{:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s, lf)


    '''
        @requires video_fps == 25, len(image_seq) >= 0, len(audio_seq) >= 0, len(image_seq) * 640 == len(audio_seq
        @modifies 
        @effects  调用syncnet 评估视频序列与音频序列是否匹配，由于syncnet以6帧为单位进行评估，输出的len(confidence) = len(image_seq) - 6
    '''
    def evaluate(self, video_fps, image_seq, audio_seq):
        if len(image_seq) <= 6:
            return None, np.array([0]), None
        offset, confidence, dists_npy = self.model.evaluate_part(video_fps, image_seq, audio_seq)
        return offset, confidence, dists_npy

    '''
        @requires confidence >= 0, start_shot >= 0, logfile == python file object
        @modifies
        @effects   根据阈值判断POI是否说话，获取起始和结束帧，格式化输出到文件
    '''
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
                slice_length = slice_end - slice_start
                logfile.writelines([self.form_convert(slice_start) + "\t" + self.form_convert(slice_length) + "\n"])
        return candidates
