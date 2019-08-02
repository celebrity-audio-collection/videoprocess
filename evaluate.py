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
        print(truelable)
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
        print("total valid  frames: ", len(valset))
        print("total missed  frames: ", len(preset - valset))
        print("ACC: ", ACC)
        print("FPR: ", FPR)
        print("Recall: ", recall)
        print("f1 micro: ", 2*ACC*recall/(recall + ACC))
    except Exception:
        print("Evaluation: divide zero")
        return -1, -1
    return FPR, recall


if __name__ == '__main__':
    evaluate_result('interview-5.csv', 'interview-5.txt')
