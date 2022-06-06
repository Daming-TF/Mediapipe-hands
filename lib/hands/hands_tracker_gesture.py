import time

from lib.hands.detector import HandDetModel
from lib.hands.hands import HandInfo, HandLandModel, HandLandModelComb
from lib.hands.hands_gesture import HandsGesture, HandsGestureLite, from_label_to_id
from lib.pose import PoseLandmark
from lib.pose import PoseLandmarkTRT
from lib.utils.draw import GestureDrawer


class HandsTrackerGesture(object):
    def __init__(self, threshold=0.5, debug=False, is_remote=False, capability=1, roi_mode=0):
        self.drawer = GestureDrawer(debug)

        self.landmark_thres = threshold
        self.roi_mode = roi_mode
        self.name = "Full" if capability == 1 else "Lite"

        self.left_hand = HandInfo("left")
        self.right_hand = HandInfo("right")

        if is_remote:
            self.detector = PoseLandmarkTRT()
        else:
            if self.roi_mode == 0:
                self.detector = HandDetModel()  # hand detector
            else:
                self.detector = PoseLandmark()  # pose landmarks

        self.hand_model = HandLandModel(
            capability=capability,
            roi_mode=self.roi_mode,
            handness_thres=self.landmark_thres,
        )  # hand landmarks
        self.gesture_model = HandsGesture(is_debug=debug)  # hand gesture recognizer

    def __call__(self, img_bgr):
        self.drawer.set_canvas(img_bgr)

        start = time.time()
        boxes, pose_landmark = self.detector(img_bgr)
        end = time.time()
        print(
            f" [!] Detector time: {(end - start) * 1000:.2f} ms. - "
            f'{"Hand-detector" if self.roi_mode == 0 else "Pose-landmarker"} model'
        )

        start = time.time()
        self.hand_model.run_with_boxes(img_bgr, boxes, self.right_hand, self.left_hand)
        end = time.time()
        print(f" [!] Landmark time: {(end - start) * 1000:.2f} ms. - {self.name} model")

        start = time.time()
        if pose_landmark is not None:
            self.gesture_model(self.left_hand, self.right_hand, pose_landmark)
        end = time.time()
        print(f" [!] Gesture time: {(end - start) * 1000:.2f} ms. - MLP model")

        for hand in [self.right_hand, self.left_hand]:
            self.drawer(hand)
        self.drawer.draw_pose_landmark(pose_landmark)

        if (self.right_hand.gesture_label is not None) or (self.left_hand.gesture_label is not None):
            is_det = True
        else:
            is_det = False

        return (
            self.drawer.get_canvas(),
            is_det,
            [self.left_hand.gesture_pred, self.right_hand.gesture_pred],
        )


class HandsTrackerGestureLite(object):
    def __init__(self, buffer_size=30, hits=5, threshold=0.5, debug=False):
        self.is_debug = debug
        self.buffer_size = buffer_size
        self.hits = hits
        self.landmark_thres = threshold
        self.name = "Lite"
        self.counter = 0

        self.left_hand = HandInfo("left")
        self.right_hand = HandInfo("right")
        self.detector = HandDetModel()  # hand detector
        self.hand_model = HandLandModel(capability=0, roi_mode=0, handness_thres=self.landmark_thres)  # hand landmarks

        target_lable_names = [
            "thumb-up",
            "rock",
            "ivu",
            "six",
            "gun",
            "ok",
            "victory",
            "heart",
            "heart-up",
            "heart-down",
            "chinese-heart",
            "korean-heart",
            "head-heart",
            "please",
            "five",
        ]
        self.target_lables = from_label_to_id(target_lable_names)
        self.gesture_model = HandsGestureLite(
            self.buffer_size, self.hits, self.target_lables, is_debug=self.is_debug
        )  # hand gesture recognizer

        self.drawer = GestureDrawer(
            self.gesture_model.label_names,
            self.gesture_model.target_lables,
            debug=self.is_debug,
        )

    def run(self, img_bgr, is_print=False):
        self.drawer.set_canvas(img_bgr)

        start = time.time()
        boxes, _ = self.detector(img_bgr)
        self.drawer.set_boxes(boxes)
        end = time.time()
        if is_print:
            print(f" [!] Detector time: {(end - start) * 1000:.2f} ms. - Hand-detector model")

        start = time.time()
        self.hand_model.run_with_boxes(img_bgr, boxes, self.right_hand, self.left_hand)
        end = time.time()
        if is_print:
            print(f" [!] Landmark time: {(end - start) * 1000:.2f} ms. - {self.name} model")

        start = time.time()
        pred_label, label_name = self.gesture_model(self.left_hand, self.right_hand, counter=self.counter)
        end = time.time()
        if is_print:
            print(f" [!] Gesture time: {(end - start) * 1000:.2f} ms. - MLP model")

        for hand in [self.right_hand, self.left_hand]:
            self.drawer(hand)
        self.drawer.draw_effect(pred_label)

        self.counter = (self.counter + 1) % self.buffer_size
        return pred_label, label_name


class HandsTrackerGestureLite2(object):
    def __init__(self, buffer_size=30, hits=5, threshold=0.5, debug=False):
        self.is_debug = debug
        self.buffer_size = buffer_size
        self.hits = hits
        self.landmark_thres = threshold
        self.name = "Lite"
        self.counter = 0

        self.left_hand = HandInfo("left")
        self.right_hand = HandInfo("right")
        self.detector = HandDetModel()  # hand detector
        # self.hand_model = HandLandModelComb(capability=0, roi_mode=0, handness_thres=self.landmark_thres)
        self.hand_model = HandLandModel(capability=0, roi_mode=0, handness_thres=self.landmark_thres)

        target_lable_names = [
            "thumb-up",
            "rock",
            "ivu",
            "six",
            "gun",
            "ok",
            "victory",
            "heart",
            "heart-up",
            "heart-down",
            "chinese-heart",
            "korean-heart",
            "head-heart",
            "please",
            "five",
        ]
        self.target_lables = from_label_to_id(target_lable_names)
        self.gesture_model = HandsGestureLite(
            self.buffer_size, self.hits, self.target_lables, is_debug=self.is_debug
        )  # hand gesture recognizer

        self.drawer = GestureDrawer(
            self.gesture_model.label_names,
            self.gesture_model.target_lables,
            debug=self.is_debug,
        )

    def run(self, img_bgr, is_print=False):
        self.drawer.set_canvas(img_bgr)

        start = time.time()
        boxes, _ = self.detector(img_bgr)
        self.drawer.set_boxes(boxes)
        end = time.time()
        if is_print:
            print(f" [!] Detector time: {(end - start) * 1000:.2f} ms. - Hand-detector model")

        start = time.time()
        self.hand_model.run_with_boxes(img_bgr, boxes, self.right_hand, self.left_hand)
        end = time.time()
        if is_print:
            print(f" [!] Landmark time: {(end - start) * 1000:.2f} ms. - {self.name} model")

        start = time.time()
        pred_label, label_name = self.gesture_model(self.left_hand, self.right_hand, counter=self.counter)
        end = time.time()
        if is_print:
            print(f" [!] Gesture time: {(end - start) * 1000:.2f} ms. - MLP model")

        for hand in [self.right_hand, self.left_hand]:
            self.drawer(hand)
        # self.drawer.draw_effect(pred_label)

        self.counter = (self.counter + 1) % self.buffer_size
        return pred_label, label_name
