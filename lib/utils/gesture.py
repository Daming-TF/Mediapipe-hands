import onnxruntime
import numpy as np


class LandmarksGestureONNX(object):
    def __init__(self, thres=0.7, is_debug=False):
        self.thres = thres
        self.is_debug = is_debug

        self.input_dims = 21 * 3 * 2
        self.model_path = "lib/models/gesture.onnx"
        self.model = onnxruntime.InferenceSession(self.model_path)
        print("*" * 70)
        print(" [!] GestureModel infer-dev:%s model:%s" % (onnxruntime.get_device(), self.model_path))

        self.label_names = [
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Gun",
            "Ok",
            "Thumb-Up",
            "Thumb-Down",
            "Fist",
            "Little-Finger",
            "Middle-Finger",
            "Flick",
            "Rock",
            "Ivu",
            "Heart",
            "Double-Chin",
            "Heart-Up",
            "Heart-Down",
        ]
        # self.target_labels = [
        #     "Six",
        #     "Thumb-Up",
        #     "Heart",
        #     "Heart-Up",
        #     "Heart-Down",
        # ]

    def __call__(self, landmarks1, landmarks2=None):
        data = np.zeros((1, self.input_dims), dtype=np.float32)

        if landmarks2 is None:
            landmarks1 = landmarks1.reshape((1, -1))
            data[0, : landmarks1.shape[1]] = landmarks1
        else:
            landmarks = np.vstack([landmarks1, landmarks2])
            landmarks = landmarks.reshape((1, -1))
            data = landmarks

        output = self.model.run(None, {self.model.get_inputs()[0].name: data})[0]
        propabs = np.exp(output[0]) / sum(np.exp(output[0]))

        pred = np.argmax(propabs)
        prop = np.max(propabs)
        if prop > self.thres:
            if self.is_debug:
                label = self.label_names[pred] + f" {int(prop * 100)}%"
            else:
                label = self.label_names[pred]
                # if label not in self.target_labels:
                #     label = None
        else:
            label = None

        return label


class HandGestures(object):
    def __init__(self, landmarks):
        self.norm_landmarks = landmarks.copy()

        self.thumb_angle = 0.0

        self.thumb_state = -1
        self.index_state = -1
        self.middle_state = -1
        self.ring_state = -1
        self.little_state = -1

        self.ok_dist = 0  # 0: unknown, 1: small, -1: large
        self.rock_dist = 0  # 0: unknown, 1: small, -1: large
        self.heart_dist = 0  # 0: unknown, 1: small, -1: large
        self.fist_dist = 0  # 0: unknown, 1: smll, -1: large

        self.gesture = None

    def print_info(self):
        print(f"thumb angle: {self.thumb_angle}")

        print(f"thumb: {self.thumb_state}")
        print(f"index: {self.index_state}")
        print(f"middle: {self.middle_state}")
        print(f"ring: {self.ring_state}")
        print(f"little: {self.little_state}")


def recognize_gesture(landmarks):
    hand = HandGestures(landmarks)

    # Finger states
    # state: -1=unknown, 0=close, 1=open

    d_4_8 = distance(hand.norm_landmarks[4], hand.norm_landmarks[8])
    d_3_7 = distance(hand.norm_landmarks[3], hand.norm_landmarks[7])
    d_4_11 = distance(hand.norm_landmarks[4], hand.norm_landmarks[11])
    d_4_10 = distance(hand.norm_landmarks[4], hand.norm_landmarks[10])

    angle0 = angle(hand.norm_landmarks[0], hand.norm_landmarks[1], hand.norm_landmarks[2])
    angle1 = angle(hand.norm_landmarks[1], hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle2 = angle(hand.norm_landmarks[2], hand.norm_landmarks[3], hand.norm_landmarks[4])
    hand.thumb_angle = angle0 + angle1 + angle2

    if hand.thumb_angle > 460:
        hand.thumb_state = 1
    else:
        hand.thumb_state = 0

    # connection judge
    if d_3_7 < 0.03:
        hand.heart_dist = 1
    elif d_3_7 > 0.05:
        hand.heart_dist = -1

    if d_4_8 < 0.03:
        hand.ok_dist = 1

    if d_4_10 < 0.04:
        hand.fist_dist = 1
    elif d_4_10 > 0.06:
        hand.fist_dist = -1

    if d_4_11 < 0.04:
        hand.rock_dist = 1
    elif d_4_11 > 0.07:
        hand.rock_dist = -1

    if hand.norm_landmarks[8][1] < hand.norm_landmarks[7][1] < hand.norm_landmarks[6][1]:
        hand.index_state = 1
    elif hand.norm_landmarks[6][1] < hand.norm_landmarks[8][1]:
        hand.index_state = 0
    else:
        hand.index_state = -1

    if hand.norm_landmarks[12][1] < hand.norm_landmarks[11][1] < hand.norm_landmarks[10][1]:
        hand.middle_state = 1
    elif hand.norm_landmarks[10][1] < hand.norm_landmarks[12][1]:
        hand.middle_state = 0
    else:
        hand.middle_state = -1

    if hand.norm_landmarks[16][1] < hand.norm_landmarks[15][1] < hand.norm_landmarks[14][1]:
        hand.ring_state = 1
    elif hand.norm_landmarks[14][1] < hand.norm_landmarks[16][1]:
        hand.ring_state = 0
    else:
        hand.ring_state = -1

    if hand.norm_landmarks[20][1] < hand.norm_landmarks[19][1] < hand.norm_landmarks[18][1]:
        hand.little_state = 1
    elif hand.norm_landmarks[18][1] < hand.norm_landmarks[20][1]:
        hand.little_state = 0
    else:
        hand.little_state = -1

    # Gesture
    if (
        hand.thumb_state == 1
        and hand.index_state == 1
        and hand.middle_state == 1
        and hand.ring_state == 1
        and hand.little_state == 1
    ):
        hand.gesture = "Five"
    elif (
        hand.thumb_state == 0
        and hand.index_state == 0
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 0
        and hand.fist_dist == 1
    ):
        hand.gesture = "Fist"
    elif (
        hand.thumb_state == 1
        and hand.index_state == 0
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 0
        and hand.fist_dist == -1
    ):
        hand.gesture = "Thumb"
    elif (
        hand.thumb_state == 0
        and hand.index_state == 1
        and hand.middle_state == 1
        and hand.ring_state == 0
        and hand.little_state == 0
    ):
        hand.gesture = "Two"
    elif (
        hand.index_state == 1
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 0
        and hand.rock_dist == 1
    ):
        hand.gesture = "One"
    elif (
        hand.thumb_state == 1
        and hand.index_state == 1
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 0
        and hand.heart_dist == -1
    ):
        hand.gesture = "Gun"
    elif (
        hand.thumb_state == 1
        and hand.index_state == 1
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 0
        and hand.heart_dist == 1
    ):
        hand.gesture = "Heart"
    elif (
        hand.index_state == 1
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 1
        and hand.rock_dist == -1
    ):
        hand.gesture = "Ivu"
    elif (
        hand.index_state == 1
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 1
        and hand.rock_dist == 1
    ):
        hand.gesture = "Rock"
    elif (
        hand.thumb_state == 1
        and hand.index_state == 0
        and hand.middle_state == 0
        and hand.ring_state == 0
        and hand.little_state == 1
    ):
        hand.gesture = "Six"
    elif (
        hand.thumb_state == 1
        and hand.index_state == 1
        and hand.middle_state == 1
        and hand.ring_state == 0
        and hand.little_state == 0
    ):
        hand.gesture = "Three"
    elif (
        hand.thumb_state == 0
        and hand.index_state == 1
        and hand.middle_state == 1
        and hand.ring_state == 1
        and hand.little_state == 0
    ):
        hand.gesture = "Three"
    elif (
        hand.thumb_state == 0
        and hand.index_state == 1
        and hand.middle_state == 1
        and hand.ring_state == 1
        and hand.little_state == 1
    ):
        hand.gesture = "Four"
    elif (
        hand.index_state == 0
        and hand.middle_state == 1
        and hand.ring_state == 1
        and hand.little_state == 1
        and hand.ok_dist == 1
    ):
        hand.gesture = "OK"
    else:
        hand.gesture = None

    # hand.print_info()

    return hand.gesture


def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a - b)


def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_ = np.arccos(cosine_angle)

    return np.degrees(angle_)
