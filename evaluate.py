import pandas as pd
import os
from common import *


def process_frame(stringin):
    strcrop = stringin.split(":")
    frame = 0
    frame += int(strcrop[0]) * 25 * 60 * 60
    frame += int(strcrop[1]) * 25 * 60
    frame += int(strcrop[2]) * 25
    frame += int(strcrop[3])
    return frame


def evaluate_result(labelfile, resultfile):
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
        print("total valid frames: ", len(valset))
        print("predict correct frames: ", len(preset & valset))
        print("\033[92mtotal missed frames: \033[0m", len(preset - valset))
        print("ACC: ", ACC)
        print("\033[92mFPR: \033[0m", FPR)
        print("\033[92mRecall: \033[0m", recall)
        print("f1 micro: ", 2 * ACC * recall / (recall + ACC))
        print("")
    except Exception:
        print("Evaluation: divide zero")
        return -1, -1
    return FPR, recall


def dataclean(output_dir):
    try:
        candidate_results = []
        with open(output_dir) as f:
            a = f.readline()
            if a != "":
                pair = a.strip().split("\t")
                candidate_results.append([pair[0], pair[1]])
            a = f.readline()
            while a != "":
                pair = a.strip().split("\t")
                slice_start = process_frame(pair[0])
                if slice_start - process_frame(candidate_results[-1][0]) - process_frame(candidate_results[-1][1]) <= 5:
                    [last_start, last_length] = candidate_results.pop()
                    new_end = slice_start + process_frame(pair[1])
                    new_length = new_end - process_frame(last_start)
                    candidate_results.append([last_start, form_convert(new_length)])
                else:
                    candidate_results.append([pair[0], pair[1]])
                a = f.readline()

        print("\033[94mstart to clean data..\033[0m")

        with open(output_dir, "w") as f:
            for pair in candidate_results:
                f.write(pair[0] + "\t" + pair[1] + "\n")

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
    dataclean("/work4/chengsitong/cslt/zzy/videoprocess/result/白宇/entertain/entertain-1.txt")
