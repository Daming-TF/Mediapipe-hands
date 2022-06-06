import onnxruntime
import numpy as np

from lib.utils.buffer import BufferPipe
from lib.utils.gesture import HandGestures, distance, angle
from lib.utils.utils import bb_iou

LABEL_NAMES = [
    "One",  # 0
    "Victory",  # 1
    "Three",  # 2
    "Four",  # 3
    "Five",  # 4
    "Six",  # 5
    "Gun",  # 6
    "Ok",  # 7
    "Thumb-Up",  # 8
    "Thumb-Down",  # 9
    "Fist",  # 10
    "Little-Finger",  # 11
    "Middle-Finger",  # 12
    "Flick",  # 13
    "Rock",  # 14
    "Ivu",  # 15
    "Heart",  # 16
    "Palm-up",  # 17
    "Three-2",  # 18
    "Others",  # 19
    "Flick-Ok",  # 20
    "Cross",  # 21
    "Stop",  # 22
    "Double-Chin",  # 23
    "Heart-Up",  # 24
    "Heart-Down",  # 25
    "Please",  # 26
    "Chinese-Heart",  # 27
    "Korean-Heart",  # 28
    "Head-Heart",  # 29
    "Others",  # 30
    "House",  # 31
    "Finger-Heart-Down",  # 32
]


class HandsGesture(object):
    def __init__(self, size, hits, target_labels, thres=0.8, is_debug=False):
        self.thres = thres
        self.is_debug = is_debug

        self.input_dims = 21 * 3 * 2
        self.one_hand_model_path = "lib/models/one_hand.onnx"
        self.two_hands_model_path = "lib/models/two_hands.onnx"
        self.one_hand_model = onnxruntime.InferenceSession(self.one_hand_model_path)
        self.two_hands_model = onnxruntime.InferenceSession(self.two_hands_model_path)
        print("*" * 100)
        print(
            f" [!] GestureModel infer-dev:{onnxruntime.get_device()} model: {self.one_hand_model_path}\n"
            f" [!] GestureModel infer-dev:{onnxruntime.get_device()} model: {self.two_hands_model_path}"
        )
        print("*" * 100)

        self.label_names = LABEL_NAMES
        self.num_one_hand_classes = self.one_hand_model.get_outputs()[0].shape[1]
        self.num_two_hands_classes = self.two_hands_model.get_outputs()[0].shape[1]
        self.target_labels = target_labels

        self.buffer = BufferPipe(
            size,
            hits,
            target_labels=self.target_labels,
            num_classes=len(self.label_names),
        )

    def run(self, landmarks1, landmarks2=None):
        data = np.zeros((1, self.input_dims), dtype=np.float32)

        if landmarks2 is None:
            landmarks1 = landmarks1.reshape((1, -1))
            data[0, : landmarks1.shape[1]] = landmarks1
            one_hand_outputs = self.one_hand_model.run(None, {self.one_hand_model.get_inputs()[0].name: data})[0]
            pred, label = self.get_final_pred(one_hand_outputs, label_offset=0)
        else:
            landmarks = np.vstack([landmarks2, landmarks1])  # may be there are some bugs
            landmarks = landmarks.reshape((1, -1))
            data = landmarks
            two_hands_outputs = self.two_hands_model.run(None, {self.two_hands_model.get_inputs()[0].name: data})[0]
            pred, label = self.get_final_pred(two_hands_outputs, label_offset=self.num_one_hand_classes)

        return pred, label

    def get_final_pred(self, outputs, label_offset=0):
        propabs = np.exp(outputs[0]) / sum(np.exp(outputs[0]))
        pred = np.argmax(propabs) + label_offset
        prop = np.max(propabs)

        if prop > self.thres:
            if pred in self.target_lables:
                pass
            else:
                pred = -1
        else:
            pred = -1

        if pred == -1:
            label = None
        else:
            label = self.label_names[pred]

        return pred, label

    def __call__(self, left_hand, right_hand, counter, pose_landmark=None):
        is_recognized = False
        left_act, right_act = self.check_wrist_position(pose_landmark)

        if (
            (left_hand.world_landmark is not None)
            and (right_hand.world_landmark is not None)
            and left_act
            and right_act
        ):

            dist = np.linalg.norm(np.mean(left_hand.landmark, axis=0) - np.mean(right_hand.landmark, axis=0))
            diagonal = np.linalg.norm(pose_landmark[0][5] - pose_landmark[0][6])
            # print(f'dist: {dist}, thres: {0.5 * diagonal}')
            if dist < 0.5 * diagonal:
                left_hand.gesture_pred, left_hand.gesture_label = self.run(
                    left_hand.world_landmark, right_hand.world_landmark
                )
                is_recognized = True

        if not is_recognized:
            for hand, is_act in zip([left_hand, right_hand], [left_act, right_act]):
                if (hand.world_landmark is not None) and is_act:
                    # world_dist = 1.0
                    # if hand.pre_world_landmark is not None:
                    #     # just recognizing the hand-pose not change relatively
                    #     world_dist = np.linalg.norm(hand.world_landmark - hand.pre_world_landmark)

                    is_det = self.recognize_candidates(hand.unprojected_world_landmark)
                    # print(f'world_dist: {world_dist}')
                    # if is_det and (world_dist < 0.015):
                    #     if world_dist < 0.01:
                    #         hand.gesture_label, hand.gesture_pred = self.run(hand.world_landmark)
                    if is_det:
                        hand.gesture_pred, hand.gesture_label = self.run(hand.world_landmark)
                    # hand.gesture_label = self.run(hand.unprojected_world_landmark)

        pred_label = self.buffer.add(preds=[left_hand.gesture_pred, right_hand.gesture_pred], counter=counter)
        if pred_label == -1:
            label_name = None
        else:
            label_name = self.label_names[pred_label]
        return pred_label, label_name

    @staticmethod
    def check_wrist_position(pose_landmark):
        # recognizing gestures when the wrist upper half of the shoulder and hip
        left_hip_y = pose_landmark[0][11][1]
        right_hip_y = pose_landmark[0][12][1]
        left_shoulder_y = pose_landmark[0][5][1]
        right_shoulder_y = pose_landmark[0][6][1]

        # left_wrist_y = pose_landmark[0][9][1]
        # right_wrist_y = pose_landmark[0][10][1]
        left_hand_center_y = pose_landmark[0][17][1]
        right_hand_center_y = pose_landmark[0][18][1]

        left_act, right_act = 0, 0
        if left_hand_center_y < 0.6 * (left_shoulder_y + left_hip_y):
            left_act = 1

        if right_hand_center_y < 0.6 * (right_shoulder_y + right_hip_y):
            right_act = 1

        return left_act, right_act

    @staticmethod
    def recognize_candidates(landmarks):
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
        if d_3_7 < 0.04:
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
        is_det = 0
        if (
            hand.thumb_state == 1
            and hand.index_state == 0
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 0
            and hand.fist_dist == -1
        ):
            # hand.gesture = "Thumb"
            is_det = 1
        if (
            hand.thumb_state == 1
            and hand.index_state == 1
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 0
            and hand.heart_dist == 1
        ):
            # hand.gesture = "Heart"
            is_det = 1
        elif (
            hand.thumb_state == 1
            and hand.index_state == 0
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 1
        ):
            # hand.gesture = "Six"
            is_det = 1
        elif (
            hand.thumb_state == 1
            and hand.index_state == 1
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 0
            and hand.heart_dist == -1
        ):
            # hand.gesture = "Gun"
            is_det = 1
        elif (
            hand.index_state == 0
            and hand.middle_state == 1
            and hand.ring_state == 1
            and hand.little_state == 1
            and hand.ok_dist == 1
        ):
            # hand.gesture = "OK"
            is_det = 1
        elif (
            hand.index_state == 1
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 1
            and hand.rock_dist == 1
        ):
            # hand.gesture = "Rock"
            is_det = 1
        elif (
            hand.index_state == 1
            and hand.middle_state == 0
            and hand.ring_state == 0
            and hand.little_state == 1
            and hand.rock_dist == -1
        ):
            # hand.gesture = "Ivu"
            is_det = 1
        elif (
            hand.thumb_state == 0
            and hand.index_state == 1
            and hand.middle_state == 1
            and hand.ring_state == 0
            and hand.little_state == 0
        ):
            # hand.gesture = "Two"
            is_det = 1

        return is_det


class HandsGestureLite(HandsGesture):
    def __init__(self, size, hits, target_labels, thres=0.8, is_debug=False):
        super(HandsGestureLite, self).__init__(size, hits, target_labels, thres, is_debug)

        self.target_lables = target_labels
        self.two_hands_iou_thres = 0.0  # 0.1

    def __call__(self, left_hand, right_hand, counter, pose_landmark=None):
        is_recognized = False

        if (left_hand.world_landmark is not None) and (right_hand.world_landmark is not None):
            iou = bb_iou(
                left_hand.landmark_to_box(box_factor=1.1),
                right_hand.landmark_to_box(box_factor=1.1),
            )
            if iou > self.two_hands_iou_thres:
                left_hand.gesture_pred, left_hand.gesture_label = self.run(
                    left_hand.world_landmark, right_hand.world_landmark
                )
                is_recognized = True

        if not is_recognized:
            for hand in [left_hand, right_hand]:
                if hand.world_landmark is not None:
                    # is_det = self.recognize_candidates(hand.unprojected_world_landmark)

                    # if is_det:
                    if True:
                        # hand.gesture_pred, hand.gesture_label = self.run(hand.world_landmark)
                        hand.gesture_pred, hand.gesture_label = self.run(hand.unprojected_world_landmark)

        pred_labels = self.buffer.add(preds=[left_hand.gesture_pred, right_hand.gesture_pred], counter=counter)

        label_names = list()
        for pred_label in pred_labels:
            if pred_label == -1:
                label_names.append(None)
            else:
                label_names.append(self.label_names[pred_label])

        return pred_labels, label_names


def from_label_to_id(label_names):
    label_name_ids = list()
    for target_label_name in label_names:
        for i, label_name in enumerate(LABEL_NAMES):
            if label_name.lower() == target_label_name:
                label_name_ids.append(i)

    return sorted(label_name_ids)
