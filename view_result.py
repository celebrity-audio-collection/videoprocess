import cv2


def view_result(video_path, result):
    cap = cv2.VideoCapture(video_path)
    if cap.get(3) > 1280:
        need_to_resize = True
    else:
        need_to_resize = False

    canditates = []
    with open(result) as f:
        a = f.readline()
        while a != "":
            pair = a.split(":")
            canditates.append([int(pair[0]), int(pair[1])])
            a = f.readline()

    for candidate_pair in canditates:
        start_shot, end_shot = candidate_pair[0], candidate_pair[1]
        cap.set(1, start_shot)
        shot_count = candidate_pair[0]
        while shot_count <= candidate_pair[1]:
            if need_to_resize:
                success, raw_image = cap.read()
                if not success:
                    break
                raw_image = cv2.resize(raw_image, (1280, 720))
            else:
                success, raw_image = cap.read()
            if not success:
                break
            cv2.imshow('Result', raw_image)
            shot_count += 1
            cv2.waitKey(20)


if __name__ == '__main__':
    view_result('videos/白宇/interview/interview-5.mp4', 'interview-5.txt')
