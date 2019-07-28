import pandas as pd


def process_frame(video_fps, stringin):
    strcrop = stringin.split(":")
    frame = 0
    frame += int(strcrop[0]) * 25 * 60 * 60
    frame += int(strcrop[1]) * 25 * 60
    frame += int(strcrop[2]) * 25
    frame += int(strcrop[3])
    frame = int(frame * video_fps / 25.0)
    return frame


def evaluate_result():
    truelable = pd.read_csv("videos/蔡依林-2.csv", encoding="utf-16-le", sep='\t')
    truelable = truelable[["入点", "出点"]].values
    print(truelable)
    canditates = []
    with open("testans.txt") as f:
        a = f.readline()
        while a != "":
            pair = a.split(":")
            canditates.append([int(pair[0]), int(pair[1])])
            a = f.readline()

    print(canditates)
    for row_index in range(len(truelable)):
        for col_index in range(len(truelable[row_index])):
            truelable[row_index][col_index] = process_frame(25, truelable[row_index][col_index])
    print(truelable)
    missed = 0
    total = 0
    preset = []
    valset = []
    for pair in canditates:
        preset += [i for i in range(pair[0], pair[1])]

    for pair in truelable:
        valset += [i for i in range(pair[0], pair[1])]

    print(valset)
    print(preset)
    preset = set(preset)
    valset = set(valset)

    # missed = len(preset - valset)/len(preset)
    # print("total valid  frames: ", len(valset))
    # print("total predictions  frames: ", len(preset))
    # print("total missed  frames: ", len(preset - valset))
    # print("miss rate: ",missed)
    FPR = len(preset - valset) / len(preset)
    recall = len(preset & valset) / len(valset)
    print("total valid  frames: ", total)
    print("total missed  frames: ", missed)
    print("FPR: ", FPR)
    print("Recall: ", recall)


if __name__ == '__main__':
    evaluate_result()
