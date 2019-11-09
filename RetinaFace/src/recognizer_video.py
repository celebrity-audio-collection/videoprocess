import sys

sys.path.append('../insightface')
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
# from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os
from RetinaFace.retinaface import RetinaFace

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
                help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
                help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/video_test.mp4",
                help='Path to output video')
ap.add_argument("--video-in", default="../datasets/videos_input/GOT_actor.mp4")

ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/mobilenet/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize detector
thresh = 0.8
gpuid = 0
scales = [360, 640]
detector = RetinaFace('../insightface/RetinaFace/model/mnet.25/mnet.25', 0, gpuid, 'net3')
cap = cv2.VideoCapture(args.video_in)
success, frame = cap.read()
im_shape = frame.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
# im_scale = 1.0
# if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

# Initialize faces embedding model
embedding_model = face_model.FaceModel(args)

# Load the classifier model
model = load_model('outputs/my_model.h5')


# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


# Initialize some useful arguments
cosine_threshold = 0.8
proba_threshold = 0.85
comparing_num = 5
trackers = []
texts = []
frames = 0

# Start streaming and recording
# cap = cv2.VideoCapture(args.video_in)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# save_width = 800
# save_height = int(800/frame_width*frame_height)
# video_out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('M','J','P','G'), 24, (save_width,save_height))

last_time = time.clock()

while True:
    ret, frame = cap.read()
    frames += 1
    # frame = cv2.resize(frame, (save_width, save_height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frames % 30 == 0:
        print('FPS:', int(1 / (time.clock() - last_time) * 30))
        last_time = time.clock()

    if frames % 3 == 0:
        trackers = []
        texts = []

        detect_tick = time.time()
        bboxes, landmarks0 = detector.detect(frame, thresh, scales=[im_scale], do_flip=False)
        detect_tock = time.time()

        if len(bboxes) != 0:
            reco_tick = time.time()
            for bbox, landmarks in zip(bboxes, landmarks0):
                bbox = bbox.astype(np.int64)
                box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

                # bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                landmarks = np.array(
                    [landmarks[0][0], landmarks[1][0], landmarks[2][0], landmarks[3][0], landmarks[4][0],
                     landmarks[0][1], landmarks[1][1], landmarks[2][1], landmarks[3][1], landmarks[4][1]])
                landmarks = landmarks.reshape((2, 5)).T
                tt = time.time()
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))
                embedding = embedding_model.get_feature(nimg).reshape(1, -1)
                print("face validation cost:", time.time() - tt)
                text = "Unknown"

                # Predict class
                preds = model.predict(embedding)
                preds = preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(preds)
                proba = preds[j]
                # Compare this vector to source class vectors to verify it is actual belong to this class
                match_class_idx = (labels == j)
                match_class_idx = np.where(match_class_idx)[0]
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = embeddings[selected_idx]
                # Calculate cosine similarity
                cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name = le.classes_[j]
                    text = "{}".format(name)
                    # print("Recognized: {} <{:.2f}>".format(name, proba*100))
                # Start tracking
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
                texts.append(text)

                y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    else:
        for tracker, text in zip(trackers, texts):
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    # video_out.write(frame)
    # print("Faces detection time: {}s".format(detect_tock-detect_tick))
    # print("Faces recognition time: {}s".format(reco_tock-reco_tick))
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# video_out.release()
cap.release()
cv2.destroyAllWindows()
