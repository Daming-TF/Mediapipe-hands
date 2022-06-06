import onnxruntime
import time
import numpy as np

from lib.hands.detector import HandDetModel
from lib.hands.hands import BasicHeatmapHands
from lib.hands.hands_trackerv1 import HandInfoSimple
from lib.pose import PoseLandmark
from lib.utils.draw import Drawerv1
from lib.utils.utils import bb_iou


class Handsv1(BasicHeatmapHands):
    def __init__(self, roi_mode, img_size=128):
        super(Handsv1, self).__init__(img_size)

        self.model_path = "./lib/models/hrnet_128x128_pose_hrnet_1_16_bright_v2-2_m00024.onnx"

        self.roi_mode = roi_mode
        self.img_size = img_size
        self.num_joints = 21
        self.dim = 3

        # Load ONXX model
        self.model = onnxruntime.InferenceSession(self.model_path)
        print("*" * 100)
        print(f"HandLandModel infer-dev:{onnxruntime.get_device()} model:{self.model_path}")
        print("*" * 100)

    def run(self, img_bgr, is_bgr=True):
        model_input, res_factor = self.pre_process(img_bgr, is_bgr)
        heatmaps = self.model.run(None, {self.model.get_inputs()[0].name: model_input})[0]
        landmarks, confs = self.post_process(heatmaps, res_factor)
        # landmarks shape: [1, 21, 2]
        return landmarks[0], confs

    def run_with_boxes(self, img_bgr, boxes, right_hand, left_hand, scale=1.5):
        right_hand.turn_off()
        left_hand.turn_off()

        if (right_hand.landmark is None) or (left_hand.landmark is None):
            if self.roi_mode == 0:
                left_hand, right_hand = self.handle_hand_detector_bbox(img_bgr, left_hand, right_hand, boxes, scale)
            else:
                left_hand, right_hand = self.handle_pose_landmark_bbox(img_bgr, left_hand, right_hand, boxes)

        for hand in [right_hand, left_hand]:
            if hand.get_flag():  # processed in box detector
                continue
            elif hand.landmark is None:  # no landmarks
                continue
            else:
                box = hand.landmark_to_box(box_factor=1.5)
                img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=1)
                landmark, confs = self.run(img_roi_bgr, is_bgr=True)
                landmark = self.get_global_coords(landmark, rect_roi_coord)

                is_hand = (np.sum(confs > 0.2) >= 15) and (np.mean(confs) < 1.0)
                if is_hand:  # predicted landmarks are reliable
                    self.set_hand_info(
                        hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord=rect_roi_coord,
                    )
                else:
                    hand.img_roi_bgr = img_roi_bgr.copy()
                    hand.reset()

        for hand in [left_hand, right_hand]:
            if not hand.get_flag():
                hand.reset()

    def handle_hand_detector_bbox(self, img_bgr, left_hand, right_hand, boxes, scale=1.5):
        candi_box = list()
        for box in boxes:
            # if right_hand.landmark is not None:
            #     iou = bb_iou(box, right_hand.landmark_to_box(box_factor=1.0))
            #     print(f'right_hand iou: {iou}')
            #
            # if left_hand.landmark is not None:
            #     iou = bb_iou(box, left_hand.landmark_to_box(box_factor=1.0))
            #     print(f'left_hand iou: {iou}')

            if (right_hand.landmark is not None) and (bb_iou(box, right_hand.landmark_to_box(box_factor=1.0)) > 0.2):
                continue
            elif (left_hand.landmark is not None) and (bb_iou(box, left_hand.landmark_to_box(box_factor=1.0)) > 0.2):
                continue
            else:
                candi_box.append(box)

        for box in candi_box:
            img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=scale)
            landmark, confs = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord)

            is_hand = (np.sum(confs > 0.2) >= 15) and (np.mean(confs) < 1.0)
            if is_hand:  # predicted landmarks are reliable
                if right_hand.landmark is None:
                    self.set_hand_info(
                        right_hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord,
                    )
                elif left_hand.landmark is None:
                    self.set_hand_info(
                        left_hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord,
                    )

        return left_hand, right_hand

    def handle_pose_landmark_bbox(self, img_bgr, left_hand, right_hand, boxes):
        candi_box = list()
        for box in boxes:
            if (right_hand.landmark is not None) and (box["type"] == "right"):
                continue
            elif (left_hand.landmark is not None) and (box["type"] == "left"):
                continue
            candi_box.append(box)

        for box in candi_box:
            # print(f'box: {box}')
            img_roi_bgr, rect_roi_coord, warp_matrix = self.get_img_roi(img_bgr, box=box, scale=1)
            landmark, confs = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord, warp_matrix)

            is_hand = (np.sum(confs > 0.2) >= 15) and (np.mean(confs) < 1.0)
            if is_hand:  # predicted landmarks are reliable
                if (right_hand.landmark is None) and (box["type"] == "right"):
                    self.set_hand_info(
                        right_hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord,
                    )
                if (left_hand.landmark is None) and (box["type"] == "left"):
                    self.set_hand_info(
                        left_hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord,
                    )
            else:
                if box["type"] == "right":
                    right_hand.img_roi_bgr = img_roi_bgr.copy()
                else:
                    left_hand.img_roi_bgr = img_roi_bgr.copy()

        return left_hand, right_hand


class HandsTrackerv1(object):
    def __init__(self, threshold=0.5, roi_mode=1, debug=False):
        self.drawer = Drawerv1(debug, roi_mode)
        self.landmark_thres = threshold
        self.roi_mode = roi_mode

        self.left_hand = HandInfoSimple("left")
        self.right_hand = HandInfoSimple("right")

        if self.roi_mode == 0:
            self.detector = HandDetModel()  # hand detector
        elif self.roi_mode == 1:
            self.detector = PoseLandmark()  # pose landmark
        else:
            raise Exception(" [!] ROI mode only support 0 or 1!")

        self.hand_model = Handsv1(self.roi_mode)

    def __call__(self, img_bgr, counter):
        self.drawer.set_canvas(img_bgr)

        # if two hands nearlly overlapped
        if (
            (self.left_hand.landmark is not None)
            and (self.right_hand.landmark is not None)
            and (bb_iou(self.left_hand.landmark_to_box(), self.right_hand.landmark_to_box()) >= 0.98)
        ):
            self.left_hand.reset()
            self.right_hand.reset()

        boxes = []
        pose_landmark = None
        if (self.left_hand.landmark is None) or (self.right_hand.landmark is None):
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
        print(f" [!] Landmark time: {(end - start) * 1000:.2f} ms. - small model")

        for hand in [self.right_hand, self.left_hand]:
            self.drawer(hand)
        self.drawer.draw_pose_landmark(pose_landmark)

        return self.drawer.get_canvas()
