# USAGE
# python speaking_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python speaking_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np


def is_speaking(prev_img, curr_img, debug=False, threshold=500, width=400, height=400):
    """
    Args:
        prev_img:
        curr_img:
    Returns:
        Bool value if a person is speaking or not
    """
    prev_img = cv2.resize(prev_img, (width, height))
    curr_img = cv2.resize(curr_img, (width, height))

    diff = cv2.absdiff(prev_img, curr_img)
    norm = np.sum(diff) / (width*height) * 100
    if debug:
        print(norm)
    return norm > threshold


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='./models/shape_predictor_68_face_landmarks.dat',
    help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-t", "--threshold", type=int, default=500,
        help="threshold of speaking or not")
ap.add_argument("-d", "--debug", action='store_true')
ap.add_argument("-w", "--width", type=int, default=800,
        help="width of window")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indices of the facial landmarks for mouth
m_start, m_end = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

prev_mouth_img = None
i = 0
margin = 10
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=args["width"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth_shape = shape[m_start:m_end+1]

        leftmost_x = min(x for x, y in mouth_shape) - margin
        bottom_y = min(y for x, y in mouth_shape) - margin
        rightmost_x = max(x for x, y in mouth_shape) + margin
        top_y = max(y for x, y in mouth_shape) + margin

        w = rightmost_x - leftmost_x
        h = top_y - bottom_y

        x = int(leftmost_x - 0.1 * w)
        y = int(bottom_y - 0.1 * h)

        w = int(1.2 * w)
        h = int(1.2 * h)

        mouth_img = gray[bottom_y:top_y, leftmost_x:rightmost_x]

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in mouth_shape:
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # confer this
        # https://github.com/seanexplode/LipReader/blob/master/TrackFaces.c#L68
        if prev_mouth_img is None:
            prev_mouth_img = mouth_img
        if is_speaking(prev_mouth_img, mouth_img, threshold=args['threshold'],
                                debug=args['debug']):
            print(str(i), "speaking")
            i += 1

        prev_mouth_img = mouth_img
        
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
