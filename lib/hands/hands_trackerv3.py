import cv2
import math
import onnxruntime
import time
import numpy as np

from lib.hands.detector import HandDetModel
from lib.hands.hands import BasicHeatmapHands, NormalizedRect
from lib.pose import PoseLandmark
from lib.utils.draw import Drawerv1
from lib.utils.utils import (
    get_rotation_matrix,
    get_translation_matrix,
    bb_iou,
)


class HandInfoSimpleV3(object):
    def __init__(self, hand_type="left", img_size=160):
        self.type = hand_type
        self.flag = False
        self.pre_landmark = None
        self.landmark = None
        self.confs = None

        self.img_size = img_size
        self.num_joints_subset = 11
        self.landmark_subset = None
        self.kWristJoint = 0
        self.kIndexFingerPIPJoint = 4
        self.kMiddleFingerPIPJoint = 6
        self.kRingFingerPIPJoint = 8
        self.kTargetAngle = math.pi * 0.5  # 90 degree represented in radian
        self.shift_x = 0.0
        self.shift_y = -0.2  # -0.1 || -0.5
        self.scale_x = 2.1  # 2.0 || 2.6
        self.scale_y = 2.1  # 2.0 || 2.6
        self.square_long = True
        self.warp_matrix = None
        self.rect = None
        self.rect_roi_coord = None
        self.img_roi_bgr = None

    def landmark_to_box(self, box_factor=1.5):
        coord_min = np.min(self.landmark, axis=0)
        coord_max = np.max(self.landmark, axis=0)
        box_c = (coord_max + coord_min) / 2
        box_size = np.max(coord_max - coord_min) * box_factor

        x_left = int(box_c[0] - box_size / 2)
        y_top = int(box_c[1] - box_size / 2)
        x_right = int(box_c[0] + box_size / 2)
        y_bottom = int(box_c[1] + box_size / 2)

        box = [(x_left, y_top), (x_right, y_bottom)]
        return box

    def get_global_coords(self, landmark):
        if self.warp_matrix is None:
            new_landmark = np.zeros_like(landmark)
            new_landmark[:, 0] = landmark[:, 0] + self.rect_roi_coord[0, 0]
            new_landmark[:, 1] = landmark[:, 1] + self.rect_roi_coord[0, 1]
        else:
            inv_warp_matrix = np.linalg.inv(self.warp_matrix)
            landmarks2d = cv2.perspectiveTransform(landmark[None, :, :2], inv_warp_matrix)[0]

            new_landmark = np.zeros_like(landmark)
            new_landmark[:, :2] = landmarks2d.copy()

        return new_landmark

    def get_rotated_img_roi(self, img_bgr):
        rect = self.normalized_landmarks_list_to_rect(img_size=(img_bgr.shape[1], img_bgr.shape[0]))
        self.rect = self.rect_transformation(rect, img_width=img_bgr.shape[1], img_height=img_bgr.shape[0])
        img_roi_bgr = self.get_rotated_rect_roi(img_bgr)
        return img_roi_bgr

    def get_rotated_rect_roi(self, img):
        img_h, img_w = img.shape[0], img.shape[1]
        x_center = int(self.rect.x_center * img_w)
        y_center = int(self.rect.y_center * img_h)
        height = int(self.rect.height * img_h)
        half = 0.5 * height
        rotation_radian = -1.0 * self.rect.rotation

        rotate_matrix = get_rotation_matrix(rotation_radian)
        translation_matrix = get_translation_matrix(-x_center, -y_center)
        coords = np.array(
            [
                [x_center - half, x_center + half, x_center - half, x_center + half],
                [y_center - half, y_center - half, y_center + half, y_center + half],
                [1, 1, 1, 1],
            ]
        )

        # rotate * translation * coordinates
        result = np.matmul(rotate_matrix, np.matmul(translation_matrix, coords))

        pt1 = (int(result[0, 0] + x_center), int(result[1, 0] + y_center))
        pt2 = (int(result[0, 1] + x_center), int(result[1, 1] + y_center))
        pt3 = (int(result[0, 2] + x_center), int(result[1, 2] + y_center))
        pt4 = (int(result[0, 3] + x_center), int(result[1, 3] + y_center))

        spts = np.float32(
            [
                [pt1[0], pt1[1]],  # left-top
                [pt2[0], pt2[1]],  # right-top
                [pt3[0], pt3[1]],  # left-bottom
                [pt4[0], pt4[1]],
            ]
        )  # right-bottom
        dpts = np.float32(
            [
                [0, 0],  # left-top
                [self.img_size - 1, 0],  # right-top
                [0, self.img_size - 1],  # left-bottm
                [self.img_size - 1, self.img_size - 1],
            ]
        )  # right-bottom
        self.warp_matrix = cv2.getPerspectiveTransform(spts, dpts)
        img_roi = cv2.warpPerspective(
            img,
            self.warp_matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )

        self.rect_roi_coord = spts.copy()

        return img_roi

    def rect_transformation(self, rect, img_width, img_height):
        width = rect.width
        height = rect.height
        rotation = rect.rotation

        if rotation == 0.0:
            rect.set_x_center(rect.x_center + width * self.shift_x)
            rect.set_y_center(rect.y_center + height * self.shift_y)
        else:
            x_shift = (
                img_width * width * self.shift_x * math.cos(rotation)
                - img_height * height * self.shift_y * math.sin(rotation)
            ) / img_width
            y_shift = (
                img_width * width * self.shift_x * math.sin(rotation)
                + img_height * height * self.shift_y * math.cos(rotation)
            ) / img_height

            rect.set_x_center(rect.x_center + x_shift)
            rect.set_y_center(rect.y_center + y_shift)

        if self.square_long:
            long_side = np.maximum(width * img_width, height * img_height)
            width = long_side / img_width
            height = long_side / img_height

        rect.set_width(width * self.scale_x)
        rect.set_height(height * self.scale_y)

        return rect

    def normalized_landmarks_list_to_rect(self, img_size):
        rotation = self.compute_rotation()
        revese_angle = self.normalize_radians(-rotation)

        # Find boundaries of landmarks.
        max_x = np.max(self.landmark_subset[:, 0])
        max_y = np.max(self.landmark_subset[:, 1])
        min_x = np.min(self.landmark_subset[:, 0])
        min_y = np.min(self.landmark_subset[:, 1])

        axis_aligned_center_x = (max_x + min_x) * 0.5
        axis_aligned_center_y = (max_y + min_y) * 0.5

        # Find boundaries of rotated landmarks.
        original_x = self.landmark_subset[:, 0] - axis_aligned_center_x
        original_y = self.landmark_subset[:, 1] - axis_aligned_center_y

        projected_x = original_x * math.cos(revese_angle) - original_y * math.sin(revese_angle)
        projected_y = original_x * math.sin(revese_angle) + original_y * math.cos(revese_angle)

        max_x = np.max(projected_x)
        max_y = np.max(projected_y)
        min_x = np.min(projected_x)
        min_y = np.min(projected_y)

        projected_center_x = (max_x + min_x) * 0.5
        projected_center_y = (max_y + min_y) * 0.5

        center_x = (
            projected_center_x * math.cos(rotation) - projected_center_y * math.sin(rotation) + axis_aligned_center_x
        )
        center_y = (
            projected_center_x * math.sin(rotation) + projected_center_y * math.cos(rotation) + axis_aligned_center_y
        )
        width = (max_x - min_x) / img_size[0]
        height = (max_y - min_y) / img_size[1]

        rect = NormalizedRect()
        rect.set_x_center(center_x / img_size[0])
        rect.set_y_center(center_y / img_size[1])
        rect.set_width(width)
        rect.set_height(height)
        rect.set_rotation(rotation)

        return rect

    def compute_rotation(self):
        self.landmark_subset = np.zeros((self.num_joints_subset, self.landmark.shape[1]), dtype=np.float32)
        self.landmark_subset[0:3] = self.landmark[0:3].copy()  # Wrist and thumb's two indexes
        self.landmark_subset[3:5] = self.landmark[5:7].copy()  # Index MCP & PIP
        self.landmark_subset[5:7] = self.landmark[9:11].copy()  # Middle MCP & PIP
        self.landmark_subset[7:9] = self.landmark[13:15].copy()  # Ring MCP & PIP
        self.landmark_subset[9:11] = self.landmark[17:19].copy()  # Pinky MPC & PIP

        x0, y0 = (
            self.landmark_subset[self.kWristJoint][0],
            self.landmark_subset[self.kWristJoint][1],
        )

        x1 = (
            self.landmark_subset[self.kIndexFingerPIPJoint][0] + self.landmark_subset[self.kRingFingerPIPJoint][0]
        ) * 0.5
        y1 = (
            self.landmark_subset[self.kIndexFingerPIPJoint][1] + self.landmark_subset[self.kRingFingerPIPJoint][1]
        ) * 0.5
        x1 = (x1 + self.landmark_subset[self.kMiddleFingerPIPJoint][0]) * 0.5
        y1 = (y1 + self.landmark_subset[self.kMiddleFingerPIPJoint][1]) * 0.5

        rotation = self.normalize_radians(self.kTargetAngle - math.atan2(-(y1 - y0), x1 - x0))
        return rotation

    @staticmethod
    def normalize_radians(angle):
        return angle - 2 * math.pi * np.floor((angle - (-math.pi)) / (2 * math.pi))

    def turn_on(self):
        self.flag = True

    def turn_off(self):
        self.flag = False

    def get_flag(self):
        return self.flag

    def reset(self):
        self.pre_landmark = None
        self.landmark = None
        self.rect_roi_coord = None
        self.landmark_subset = None
        self.warp_matrix = None
        self.rect = None
        self.confs = None


class Handsv3(BasicHeatmapHands):
    def __init__(self, roi_mode, img_size=160):
        super(Handsv3, self).__init__(img_size)

        self.model_path = "./lib/models/mobilenetv2_160x160_alpha_140_coarserefine_align_data_v2_4_try02.onnx"
        self.roi_mode = roi_mode
        self.img_size = img_size
        self.num_joints = 21
        self.dim = 3

        # Load ONXX model
        self.model = onnxruntime.InferenceSession(self.model_path)
        print("*" * 70)
        print(f"HandLandModel infer-dev:{onnxruntime.get_device()} model:{self.model_path}")

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
                img_roi_bgr = hand.get_rotated_img_roi(img_bgr)
                landmark, confs = self.run(img_roi_bgr, is_bgr=True)
                landmark = hand.get_global_coords(landmark)

                print(f"np.sum(confs > 0.2): {np.sum(confs > 0.2)}")
                is_hand = np.sum(confs > 0.2) >= 15
                if is_hand:  # predicted landmarks are reliable
                    self.set_hand_info(
                        hand,
                        landmark,
                        img_roi_bgr,
                        confs,
                        rect_roi_coord=hand.rect_roi_coord,
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
            if right_hand.landmark is not None:
                iou = bb_iou(box, right_hand.landmark_to_box(box_factor=1.0))
                print(f"Right iou: {iou}")

            if left_hand.landmark is not None:
                iou = bb_iou(box, left_hand.landmark_to_box(box_factor=1.0))
                print(f"Left iou: {iou}")

            if (right_hand.landmark is not None) and (bb_iou(box, right_hand.landmark_to_box(box_factor=1.0)) > 0.3):
                continue
            if (left_hand.landmark is not None) and (bb_iou(box, left_hand.landmark_to_box(box_factor=1.0)) > 0.3):
                continue
            candi_box.append(box)

        for box in candi_box:
            img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=scale)
            landmark, confs = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord)

            is_hand = np.sum(confs > 0.2) >= 11
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

            is_hand = np.sum(confs > 0.2) >= 11
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


class HandsTrackerv3(object):
    def __init__(self, img_size=160, threshold=0.5, roi_mode=1, debug=False):
        self.img_size = img_size
        self.drawer = Drawerv1(debug, roi_mode)
        self.landmark_thres = threshold
        self.roi_mode = roi_mode

        self.left_hand = HandInfoSimpleV3("left", self.img_size)
        self.right_hand = HandInfoSimpleV3("right", self.img_size)

        if self.roi_mode == 0:
            self.detector = HandDetModel()  # hand detector
        elif self.roi_mode == 1:
            self.detector = PoseLandmark()  # pose landmark
        else:
            raise Exception(" [!] ROI mode only support 0 or 1!")

        self.hand_model = Handsv3(self.roi_mode)

    def __call__(self, img_bgr, counter):
        self.drawer.set_canvas(img_bgr)

        # if two hands nearlly overlapped
        if (
            (self.left_hand.landmark is not None)
            and (self.right_hand.landmark is not None)
            and (bb_iou(self.left_hand.landmark_to_box(), self.right_hand.landmark_to_box()) >= 0.8)
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
