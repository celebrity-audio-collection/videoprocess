import pandas as pd
import os
from common import *
import numpy as np
import time
import cv2


def process_frame(stringin):
    if stringin == "":
        return
    strcrop = stringin.split(":")
    frame = 0
    frame += int(strcrop[0]) * 25 * 60 * 60
    frame += int(strcrop[1]) * 25 * 60
    frame += int(strcrop[2]) * 25
    frame += int(strcrop[3])
    return frame


def evaluate_result(labelfile, resultfile, video_total_frame):
    log_dir = os.path.join(config.log_dir, "evaluate.txt")
    if not os.path.exists(log_dir):
        with open(log_dir, "w") as f:
            print("create evaluate.txt")

    video_type = resultfile.split("/")[-2]
    if video_type == "entertainment":
        video_type = "entertain"
    if video_type == "tv":
        video_type = "tvs"

    try:
        # truelable = pd.read_csv(os.path.join(os.getcwd(), 'videos', POI, POI + "-" + str(config.video_num) + '.csv'),
        #                         encoding="utf-16-le", sep='\t')
        truelable = pd.read_csv(labelfile, encoding="utf-16-le", sep='\t')
        truelable = truelable[["入点", "出点"]].values
        # print(truelable)
    except Exception:
        print("Evaluation: the labels file does not exist.")
        return -1, -1

    try:
        canditates = []
        with open(resultfile) as f:
            a = f.readline()
            while a != "":
                pair = a.split("\t")
                slice_start = process_frame(pair[0])
                slice_end = slice_start + process_frame(pair[1])
                canditates.append([slice_start, slice_end])
                # canditates.append([int(pair[0]), int(pair[1])])
                a = f.readline()
    except Exception:
        print("Evaluation: the result file does not exist.")
        return -1, -1
    # print(canditates)
    for row_index in range(len(truelable)):
        for col_index in range(len(truelable[row_index])):
            truelable[row_index][col_index] = process_frame(truelable[row_index][col_index])
    # print(truelable)
    missed = 0
    total = 0
    preset = []
    valset = []
    for pair in canditates:
        preset += [i for i in range(pair[0], pair[1])]

    for pair in truelable:
        valset += [i for i in range(pair[0], pair[1])]

    # print(valset)
    # print(preset)
    preset = set(preset)
    valset = set(valset)

    # missed = len(preset - valset)/len(preset)
    # print("total valid  frames: ", len(valset))
    # print("total predictions  frames: ", len(preset))
    # print("total missed  frames: ", len(preset - valset))
    # print("miss rate: ",missed)]
    try:
        ACC = len(preset & valset) / len(preset)
        FPR = len(preset - valset) / len(preset)
        recall = len(preset & valset) / len(valset)
        proportion = len(preset) / int(video_total_frame)  # 得到帧数占视频总帧数的百分比
        print("total valid frames: ", len(valset))
        print("predict correct frames: ", len(preset & valset))
        print("\033[92mtotal missed frames: \033[0m", len(preset - valset))
        print("\033[92mFPR: \033[0m", FPR)
        print("\033[92mRecall: \033[0m", recall)
        print("\033[92mProportion: \033[0m", proportion)
        print("f1 micro: ", 2 * ACC * recall / (recall + ACC))
        print("")

        # 计算均值
        current_log = []
        with open(log_dir) as f:
            line = f.readline()
            while line != "":
                pair = line.strip().split("\t")
                current_log.append([pair[0], pair[1], pair[2], pair[3], pair[4]])
                line = f.readline()

        tag = False

        with open(log_dir, "w") as f:
            for log_type in current_log:
                if log_type[0] == video_type:
                    tag = True
                    num = int(1 + int(log_type[4]))
                    new_FPR = ((num - 1) * float(log_type[1]) + round(FPR, 3)) / num
                    new_recall = ((num - 1) * float(log_type[2]) + round(recall, 3)) / num
                    new_proportion = ((num - 1) * float(log_type[3]) + round(proportion, 3)) / num
                    log_type[0] = video_type
                    log_type[1] = str(new_FPR)
                    log_type[2] = str(new_recall)
                    log_type[3] = str(new_proportion)
                    log_type[4] = str(num)
                f.write(log_type[0] + '\t' + log_type[1] + '\t' + log_type[2] + '\t' + log_type[3] + '\t' + log_type[
                    4] + '\n')

            if not tag:
                f.write(video_type + '\t' + str(round(FPR, 3)) + '\t' + str(round(recall, 3)) + '\t' +
                        str(round(proportion, 3)) + '\t' + "1" + '\n')
    except Exception:
        print("Evaluation: divide zero")
        return -1, -1

    return FPR, recall


def dataclean(output_dir, video_total_frame):
    try:
        print("\033[94mstart to clean data..\033[0m")
        # 将syncnet得到的结果导入，进行简单的数据清洗
        sync_results = []  # 存入的是[开始帧，结束帧]
        with open(output_dir) as f:
            a = f.readline()
            if a != "":
                pair = a.strip().split("\t")
                sync_results.append([process_frame(pair[0]), process_frame(pair[0]) + process_frame(pair[1])])
                a = f.readline()

                while a != "":
                    pair = a.strip().split("\t")
                    slice_start = process_frame(pair[0])
                    if slice_start - sync_results[-1][1] <= 10:
                        [last_start, last_end] = sync_results.pop()
                        new_end = slice_start + process_frame(pair[1])
                        sync_results.append([last_start, new_end])
                    else:
                        sync_results.append([slice_start, slice_start + process_frame(pair[1])])
                    a = f.readline()

        # 导入声纹识别的数据
        wav_results = []
        with open(output_dir.replace(config.output_dir, config.wav_output_dir)) as f:
            a = f.readline()
            i = -1
            while a != "":
                if a[0] == "=":
                    wav_results.append([])
                    i += 1
                else:
                    pair = a.strip().split('\t')
                    wav_results[i].append([process_frame(pair[0]), process_frame(pair[1])])
                a = f.readline()

        # 声纹识别得到的片段=>帧集合
        wav_sets = []
        for wav_person in wav_results:
            wav_set = []
            for pair in wav_person:
                wav_set += [i for i in range(pair[0], pair[1])]
            wav_set = set(wav_set)
            wav_sets.append(wav_set)

        # 清洗后的syncnet片段=>集合
        sync_set = []
        for pair in sync_results:
            sync_set += [i for i in range(pair[0], pair[1])]
        sync_set = set(sync_set)

        # 判断声纹识别出的人物中哪些人是POI
        good_person = []
        for i in range(len(wav_sets)):
            print("id: %s  length: %s 交: %s proportion: %s" % (
            i,len(wav_sets[i]), len(sync_set & wav_sets[i]), len(sync_set & wav_sets[i]) / len(wav_sets[i])))
            if len(sync_set & wav_sets[i]) / len(wav_sets[i]) >= config.poi_choose_threshold:
                good_person.append([i, len(sync_set & wav_sets[i])])

        poi_num = config.poi_num
        # 没有很好的POI
        if len(good_person) == 0:
            print("\033[94mHere is no good poi to select. Model cannot get a good result.\033[0m")
            for i in range(len(wav_sets)):
                good_person.append([i, len(sync_set & wav_sets[i])])
            poi_num = 1

        good_person = np.array(good_person)
        good_person = good_person[np.lexsort(-good_person.T)]
        [people_num, x] = good_person.shape

        # 将声纹片段中和syncnet交集大于{config.segment_choose_threshold}的片段作为结果
        real_result = []
        i = -1
        for id in range(people_num):
            i += 1
            if i >= poi_num:  # 认为几个片段是主人公
                break
            person = good_person[id]
            print("\033[94mwav poi:\033[0m %s : \033[94m %s frame\033[0m" % (person[0], person[1]))
            for wav_segment in wav_results[person[0]]:  # wav_segment: 声纹得到的声音片段
                wav_start = wav_segment[0]
                wav_end = wav_segment[1]
                wav_segment_set = set([i for i in range(wav_start, wav_end)])
                if len(wav_segment_set & sync_set) / len(wav_segment_set) >= config.segment_choose_threshold:
                    if wav_end - wav_start < 40:
                        trim_times = 0
                    elif wav_end - wav_start >= 40:
                        trim_times = 1
                    elif wav_end - wav_start >= 150:
                        trim_times = 2
                    else:
                        trim_times = 0
                    real_result.append([wav_start + config.trim_frame * trim_times,
                                        wav_end - wav_start - config.trim_frame * trim_times])
        real_result.sort()

        print(real_result)
        # 数据清洗
        candidate_results = []  # 存入的是[开始帧，持续帧]
        if len(real_result) != 0:
            candidate_results.append(real_result[0])
            if len(real_result) == 1:
                print("only one segment..")
                with open(output_dir, "w") as f:
                    f.write(form_convert(int(real_result[0][0])) + "\t" + form_convert(int(real_result[0][1])) + "\n")
                return
            for pair in real_result[1:]:
                slice_start = pair[0]
                if slice_start - int(candidate_results[-1][0]) - int(
                        candidate_results[-1][1]) <= config.concat_threshold:
                    [last_start, last_length] = candidate_results.pop()
                    new_end = slice_start + pair[1]
                    candidate_results.append([last_start, new_end - last_start])
                else:
                    candidate_results.append([slice_start, pair[1]])
        else:
            print("\033[91mNo segment was selected! This video will get a bad result.\033[0m")
            # os.remove(output_dir)
            return

        candidate_set = []
        # 数据写入
        with open(output_dir, "w") as f:
            for pair in candidate_results:
                candidate_set += [i for i in range(pair[0], pair[1])]
                f.write(form_convert(int(pair[0])) + "\t" + form_convert(int(pair[1])) + "\n")

        if len(candidate_set) / int(video_total_frame) < config.proportion:
            print("bad video, remove it")
            # os.remove(output_dir)

        print("\033[94mfile writing..\033[0m")

    except Exception:
        print("\033[91mdataclean failed\033[0m")


def form_convert(frame_id):
    h = int(frame_id / 90000)
    rest = frame_id % 90000
    m = int(rest / 1500)
    rest = rest % 1500
    s = int(rest / 25)
    lf = rest % 25
    return "{:0>2d}:{:0>2d}:{:0>2d}:{:0>2d}".format(h, m, s, lf)


if __name__ == '__main__':
    namelist = os.listdir(config.wav_output_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    log_dir = os.path.join(config.log_dir, "evaluate.txt")
    with open(log_dir, "w") as f:
        print("create evaluate.txt")
    for name in namelist:
        typedir = os.path.join(config.wav_output_dir, name)
        typelist = os.listdir(typedir)
        for type in typelist:
            for root, dirs, files in os.walk(os.path.join(typedir, type)):
                for file in files:
                    filedir = os.path.join(root, file)
                    print(filedir)
                    video_dir = (filedir.replace(config.wav_output_dir, config.video_base_dir)).replace("txt", "mp4")
                    if not os.path.exists(video_dir):
                        video_dir = (filedir.replace(config.wav_output_dir, config.video_base_dir)).replace("txt",
                                                                                                            "MP4")
                    cap = cv2.VideoCapture(video_dir)
                    video_total_frame = cap.get(7)
                    dataclean(filedir.replace(config.wav_output_dir, config.output_dir), video_total_frame)
                    evaluate_result(filedir.replace(config.wav_output_dir, config.video_base_dir).replace("txt", "csv"),
                                    filedir.replace(config.wav_output_dir, config.output_dir), video_total_frame)
