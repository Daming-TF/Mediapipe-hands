import cv2
import math
import onnxruntime
import tensorflow as tf
import numpy as np

from lib.tracker.kalman import KalmanTracker
from lib.utils.utils import (
    get_rotation_matrix,
    get_translation_matrix,
    bb_iou,
    smooth_pts,
)


class HandInfo(object):
    def __init__(self, hand_type="left"):
        self.type = hand_type
        self.gesture_label = None
        self.gesture_pred = -1
        self.flag = False
        self.pre_landmark = None
        self.landmark = None
        self.unprojected_world_landmark = None
        self.world_landmark = None
        self.pre_world_landmark = None
        self.handness = None

        self.img_size = 224
        self.num_joints_subset = 11
        self.landmark_subset = None
        self.kWristJoint = 0
        self.kIndexFingerPIPJoint = 4
        self.kMiddleFingerPIPJoint = 6
        self.kRingFingerPIPJoint = 8
        self.kTargetAngle = math.pi * 0.5  # 90 degree represented in radian
        self.shift_x = 0.0
        self.shift_y = -0.2  # -0.1 || -0.5
        self.scale_x = 2.2  # 2.2  # 2.0 || 2.6
        self.scale_y = 2.2  # 2.2  # 2.0 || 2.6
        self.square_long = True
        self.warp_matrix = None
        self.rect = None
        self.rect_roi_coord = None
        self.img_roi_bgr = None

    def world_landmark_projection(self):
        if self.world_landmark is not None:
            self.pre_world_landmark = self.world_landmark.copy()

        if self.rect:
            worldlandmark = np.zeros_like(self.unprojected_world_landmark)
            radian = self.rect.rotation
            worldlandmark[:, 0] = (
                math.cos(radian) * self.unprojected_world_landmark[:, 0]
                - math.sin(radian) * self.unprojected_world_landmark[:, 1]
            )
            worldlandmark[:, 1] = (
                math.sin(radian) * self.unprojected_world_landmark[:, 0]
                + math.cos(radian) * self.unprojected_world_landmark[:, 1]
            )
            worldlandmark[:, 2] = self.unprojected_world_landmark[:, 2].copy()
            self.world_landmark = worldlandmark

    def landmark_to_box(self, box_factor=1.0):
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
            new_landmark[:, 2] = landmark[:, -1].copy()

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
        self.gesture_label = None
        self.gesture_pred = -1
        self.world_landmark_projection()
        self.flag = True

    def turn_off(self):
        self.flag = False

    def get_flag(self):
        return self.flag

    def reset(self):
        self.pre_landmark = None
        self.landmark = None
        self.handness = None
        self.rect_roi_coord = None
        self.landmark_subset = None
        self.warp_matrix = None
        self.rect = None
        self.unprojected_world_landmark = None
        self.pre_world_landmark = None
        self.world_landmark = None
        self.gesture_label = None
        self.gesture_pred = -1
        # self.img_roi_bgr = None


class HandInfoTracker(HandInfo):
    def __init__(self, hand_type="left"):
        super(HandInfoTracker, self).__init__(hand_type=hand_type)
        self.kalman = KalmanTracker(min_hits=3)
        self.track_box = []
        self.scale_x = 2.0  # 2.2  # 2.0 || 2.6
        self.scale_y = 2.0  # 2.2  # 2.0 || 2.6

    def release_kalman(self):
        self.kalman.release()


class HandLandModel(object):
    def __init__(self, capability=1, roi_mode=0, handness_thres=0.5):
        self.roi_mode = roi_mode
        if capability == 0:
            self.model_path = "../lib/models/hand_landmark_lite.onnx"
        elif capability == 1:
            self.model_path = "../lib/models/hand_landmark_full.onnx"
        else:
            raise ValueError(" [!] Capability only supports between 0 and 1!")

        self.img_size = 224
        self.num_joints = 21
        self.dim = 3
        self.handness_thres = handness_thres
        self.righthand_prop_thres = 0.5

        # Load ONXX model
        self.model = onnxruntime.InferenceSession(self.model_path)
        print("*" * 70)
        print(f" [!] HandLandModel infer-dev: {onnxruntime.get_device()} model: {self.model_path}")

    def pre_process(self, img, is_bgr=True):
        res_factor = np.array(
            [img.shape[1] / self.img_size, img.shape[0] / self.img_size],
            dtype=np.float32,
        )
        img_res = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        if is_bgr:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_res

        img_norm = img_rgb.astype(np.float32) / 255.0
        model_input = img_norm[None, ...]
        model_input = np.transpose(model_input, [0, 3, 1, 2])  # NHWC to HCHW

        return model_input, res_factor

    def post_process(self, out, res_factor):
        out = out.reshape(self.num_joints, self.dim)
        out[:, :2] = out[:, :2] * res_factor
        return out

    def run(self, img_bgr, is_get_2d=False, is_bgr=True):
        model_input, res_factor = self.pre_process(img_bgr, is_bgr)

        outputs = self.model.run(None, {self.model.get_inputs()[0].name: model_input})
        landmarks, handness, righthand, world_landmarks = (
            outputs[0][0],
            outputs[1][0],
            outputs[2][0],
            outputs[3][0],
        )
        world_landmarks = world_landmarks.reshape(self.num_joints, self.dim)

        if is_get_2d:
            landmarks = self.post_process(landmarks, res_factor)[:, :2]
        else:
            landmarks = self.post_process(landmarks, res_factor)

        return landmarks, handness, righthand, world_landmarks

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
                landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
                landmark = hand.get_global_coords(landmark)

                if handness >= self.handness_thres:
                    # hand_info and righthand-prop inconsistent
                    if (righthand_prop >= self.righthand_prop_thres and hand.type != "right") or (
                        righthand_prop < self.righthand_prop_thres and hand.type != "left"
                    ):
                        hand.img_roi_bgr = img_roi_bgr.copy()
                        hand.reset()
                    else:
                        self.set_hand_info(
                            hand,
                            landmark,
                            world_landmark,
                            handness,
                            img_roi_bgr,
                            rect_roi_coord=hand.rect_roi_coord,
                        )
                        # hand.turn_on()
                else:
                    hand.img_roi_bgr = img_roi_bgr.copy()
                    hand.reset()

        for hand in [left_hand, right_hand]:
            if not hand.get_flag():
                hand.reset()

    def handle_hand_detector_bbox(self, img_bgr, left_hand, right_hand, boxes, scale):
        candi_box = list()
        for box in boxes:
            if (right_hand.landmark is not None) and (bb_iou(box, right_hand.landmark_to_box()) > 0.2):
                continue
            elif (left_hand.landmark is not None) and (bb_iou(box, left_hand.landmark_to_box()) > 0.2):
                continue
            else:
                candi_box.append(box)

        for box in candi_box:
            img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=scale)
            landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord)

            if handness[0] > self.handness_thres:  # predicted landmarks are reliable
                if (righthand_prop >= self.righthand_prop_thres) and (right_hand.landmark is None):
                    self.set_hand_info(
                        right_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
                        rect_roi_coord,
                    )
                if (righthand_prop < self.righthand_prop_thres) and (left_hand.landmark is None):
                    self.set_hand_info(
                        left_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
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
            img_roi_bgr, rect_roi_coord, warp_matrix = self.get_img_roi(img_bgr, box=box, scale=1)
            landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord, warp_matrix)

            if handness[0] > self.handness_thres:  # predicted landmarks are reliable
                if (
                    (righthand_prop >= self.righthand_prop_thres)
                    and (right_hand.landmark is None)
                    and (box["type"] == "right")
                ):
                    # if (right_hand.landmark is None) and (box["type"] == "right"):
                    self.set_hand_info(
                        right_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
                        rect_roi_coord,
                    )
                if (
                    (righthand_prop < self.righthand_prop_thres)
                    and (left_hand.landmark is None)
                    and (box["type"] == "left")
                ):
                    # if (left_hand.landmark is None) and (box["type"] == "left"):
                    self.set_hand_info(
                        left_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
                        rect_roi_coord,
                    )
            else:
                if box["type"] == "right":
                    right_hand.img_roi_bgr = img_roi_bgr.copy()
                else:
                    left_hand.img_roi_bgr = img_roi_bgr.copy()

        return left_hand, right_hand

    @staticmethod
    def set_hand_info(hand, landmark, world_landmark, handness, img_roi_bgr, rect_roi_coord=None):
        if hand.landmark is not None:
            hand.pre_landmark = hand.landmark.copy()
        hand.landmark = landmark.copy()

        # smoothing
        if hand.pre_landmark is not None:
            hand.landmark = smooth_pts(hand.pre_landmark, hand.landmark, rect_roi_coord)

        hand.unprojected_world_landmark = world_landmark.copy()
        hand.handness = handness[0]
        hand.img_roi_bgr = img_roi_bgr.copy()
        if rect_roi_coord is not None:
            hand.rect_roi_coord = rect_roi_coord.copy()
        hand.turn_on()

    def get_img_roi(self, img_bgr, box, scale):
        if isinstance(box, list):  # [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]
            rotation = None
            pass
        elif isinstance(box, dict):
            rotation = box["rotation"]
            box = box["box"]  # {'type': String, 'box': [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]}
        else:
            rotation = None

        ori_img_h = img_bgr.shape[0]
        ori_img_w = img_bgr.shape[1]
        box_w = box[1][0] - box[0][0]
        box_h = box[1][1] - box[0][1]
        new_box = [
            (
                max(0, box[0][0] - 0.5 * (scale - 1.0) * box_w),
                max(0, box[0][1] - 0.5 * (scale - 1.0) * box_h),
            ),
            (
                min(box[1][0] + 0.5 * (scale - 1.0) * box_w, ori_img_w),
                min(box[1][1] + 0.5 * (scale - 1.0) * box_h, ori_img_h),
            ),
        ]

        if rotation is not None:
            img_roi, rect_roi_coord, warp_matrix = self.get_rotated_rect_roi(img_bgr, new_box, rotation=rotation)
            return img_roi, rect_roi_coord, warp_matrix
        else:
            img_roi = img_bgr[
                int(new_box[0][1]) : int(new_box[1][1]),
                int(new_box[0][0]) : int(new_box[1][0]),
            ].copy()
            rect_roi_coord = np.array(
                [
                    [new_box[0][0], new_box[0][1]],  # left-top
                    [new_box[1][0], new_box[0][1]],  # right-top
                    [new_box[0][0], new_box[1][1]],  # left-bottom
                    [new_box[1][0], new_box[1][1]],
                ],
                dtype=np.float32,
            )  # right-bottom
            return img_roi, rect_roi_coord

    def get_rotated_rect_roi(self, img, box, rotation):
        x_center = int(box[0][0] + 0.5 * (box[1][0] - box[0][0]))
        y_center = int(box[0][1] + 0.5 * (box[1][1] - box[0][1]))
        height = int(box[1][1] - box[0][1])

        half = 0.5 * height
        rotation_radian = -1.0 * rotation

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
        warp_matrix = cv2.getPerspectiveTransform(spts, dpts)
        img_roi = cv2.warpPerspective(
            img,
            warp_matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )

        rect_roi_coord = spts.copy()

        return img_roi, rect_roi_coord, warp_matrix

    @staticmethod
    def get_global_coords(landmark, rect_roi_coord, warp_matrix=None):
        if warp_matrix is None:
            new_landmark = np.zeros_like(landmark)
            new_landmark[:, 0] = landmark[:, 0] + rect_roi_coord[0, 0]
            new_landmark[:, 1] = landmark[:, 1] + rect_roi_coord[0, 1]
        else:
            inv_warp_matrix = np.linalg.inv(warp_matrix)
            landmarks2d = cv2.perspectiveTransform(landmark[None, :, :2], inv_warp_matrix)[0]

            new_landmark = np.zeros_like(landmark)
            new_landmark[:, :2] = landmarks2d.copy()
            new_landmark[:, 2] = landmark[:, -1].copy()

        return new_landmark


class NormalizedRect(object):
    def __init__(self):
        self.x_center = None
        self.y_center = None
        self.width = None
        self.height = None
        self.rotation = None

    def set_x_center(self, value):
        self.x_center = value

    def set_y_center(self, value):
        self.y_center = value

    def set_width(self, value):
        self.width = value

    def set_height(self, value):
        self.height = value

    def set_rotation(self, value):
        self.rotation = value


class Hands(object):
    # Note: This warper only considers single hand
    def __init__(self, capability=1):
        if capability == 0:
            self.tflite_model_path = "./lib/models/hand_landmark_lite.tflite"
        elif capability == 1:
            self.tflite_model_path = "./lib/models/hand_landmark_full.tflite"
        else:
            raise ValueError(" [!] Capability only supports between 0 and 1!")

        self.img_size = 224
        self.num_joints = 21
        self.dim = 3
        self.img_roi_bgr = None
        self.rect_roi_coords = None
        self.unprojected_world_landmarks = None

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def clear_history(self):
        self.img_roi_bgr = None
        self.rect_roi_coords = None

    def pre_process(self, img, is_bgr=True):
        res_factor = np.array(
            [img.shape[1] / self.img_size, img.shape[0] / self.img_size],
            dtype=np.float32,
        )
        img_res = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        if is_bgr:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_res

        img_norm = img_rgb.astype(np.float32) / 255.0
        model_input = img_norm[None, ...]

        return model_input, res_factor

    def post_process(self, out, res_factor):
        out = out.reshape(self.num_joints, self.dim)
        out[:, :2] = out[:, :2] * res_factor
        return out

    def run(self, img_bgr, is_get_2d=True, is_bgr=True):
        model_input, res_factor = self.pre_process(img_bgr, is_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], model_input)

        self.interpreter.invoke()
        landmarks = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        handness = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        righthand = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        world_landmarks = self.interpreter.get_tensor(self.output_details[3]["index"])[0]

        if is_get_2d:
            landmarks = self.post_process(landmarks, res_factor)[:, :2]
        else:
            landmarks = self.post_process(landmarks, res_factor)

        return landmarks, handness, righthand, world_landmarks

    def run_with_boxes(self, img_bgr, boxes, scale=1.5):
        landmarks_output = list()
        handness_output = list()
        righthands_output = list()
        world_landmarks_output = list()
        boxes_output = list()
        rects_output = list()

        for box in boxes:
            self.img_roi_bgr = self.get_img_roi(img_bgr, box, scale)
            landmarks, handness, righthand, _ = self.run(self.img_roi_bgr, is_get_2d=True, is_bgr=True)
            landmarks = self.get_global_coords(landmarks)

            landmarks_output.append(landmarks)
            world_landmarks_output.append(None)
            handness_output.append(handness[0])
            righthands_output.append(righthand)
            boxes_output.append(self.rect_roi_coords)
            rects_output.append(None)

        return (
            landmarks_output,
            handness_output,
            righthands_output,
            boxes_output,
            rects_output,
            world_landmarks_output,
        )

    def get_img_roi(self, img_bgr, box, scale):
        ori_img_h = img_bgr.shape[0]
        ori_img_w = img_bgr.shape[1]
        box_w = box[1][0] - box[0][0]
        box_h = box[1][1] - box[0][1]
        new_box = [
            (
                max(0, box[0][0] - 0.5 * (scale - 1.0) * box_w),
                max(0, box[0][1] - 0.5 * (scale - 1.0) * box_h),
            ),
            (
                min(box[1][0] + 0.5 * (scale - 1.0) * box_w, ori_img_w),
                min(box[1][1] + 0.5 * (scale - 1.0) * box_h, ori_img_h),
            ),
        ]

        img_roi = img_bgr[
            int(new_box[0][1]) : int(new_box[1][1]),
            int(new_box[0][0]) : int(new_box[1][0]),
        ].copy()
        self.rect_roi_coords = np.float32(
            [
                [new_box[0][0], new_box[0][1]],  # left-top
                [new_box[1][0], new_box[0][1]],  # right-top
                [new_box[0][0], new_box[1][1]],  # left-bottom
                [new_box[1][0], new_box[1][1]],
            ]
        )  # right-bottom

        return img_roi

    def get_global_coords(self, landmarks):
        new_landmarks = np.zeros_like(landmarks)
        new_landmarks[:, 0] = landmarks[:, 0] + self.rect_roi_coords[0, 0]
        new_landmarks[:, 1] = landmarks[:, 1] + self.rect_roi_coords[0, 1]
        return new_landmarks


class MediapipeHands(Hands):
    # Note: This warper only considers single hand
    def __init__(self, capability=1):
        super(MediapipeHands, self).__init__(capability)

        # reference to mediapipe's varialbes and the logtics
        self.num_joints_subset = 11
        (
            self.pre_landmarks,
            self.pre_landmarks_subset,
            self.unprojected_world_landmarks,
        ) = (None, None, None)
        self.kWristJoint = 0
        self.kIndexFingerPIPJoint = 4
        self.kMiddleFingerPIPJoint = 6
        self.kRingFingerPIPJoint = 8
        self.kTargetAngle = math.pi * 0.5  # 90 degree represented in radian
        self.shift_x = 0.0
        self.shift_y = -0.2  # -0.1 || -0.5
        self.scale_x = 2.6  # 2.0 || 2.6
        self.scale_y = 2.6  # 2.0 || 2.6
        self.square_long = True
        self.warp_matrix = None

    def clear_history(self):
        self.img_roi_bgr = None
        self.rect_roi_coords = None

        self.pre_landmarks = None
        self.pre_landmarks_subset = None
        self.warp_matrix = None

    def run_with_boxes(self, img_bgr, boxes, scale=1.5):
        landmarks_output = list()
        handness_output = list()
        righthands_output = list()
        world_landmarks_output = list()
        boxes_output = list()
        rects_output = list()

        for box in boxes:
            rect = None

            if self.pre_landmarks is None:  # using predefined rect-roi as the input
                self.img_roi_bgr = self.get_img_roi(img_bgr, box, scale)
            else:  # using previous landmarks to calculate the rotated roi as the input
                rect = self.normalized_landmarks_list_to_rect(img_size=(img_bgr.shape[1], img_bgr.shape[0]))
                rect = self.rect_transformation(rect, img_width=img_bgr.shape[1], img_height=img_bgr.shape[0])
                self.img_roi_bgr = self.get_rotated_rect_roi(img_bgr, rect)

            landmarks, handness, righthand, world_landmarks = self.run(self.img_roi_bgr, is_get_2d=False, is_bgr=True)
            landmarks = self.get_global_coords(landmarks)
            self.pre_landmarks = landmarks.copy()  # TODO: Need to consider two hands, at now only consider one hand

            world_landmarks = world_landmarks.reshape(self.num_joints, self.dim)
            self.unprojected_world_landmarks = world_landmarks.copy()

            if rect is not None:
                world_landmarks = self.world_landmark_projection(world_landmarks, rect)

            landmarks_output.append(landmarks)
            world_landmarks_output.append(world_landmarks)
            handness_output.append(handness[0])
            righthands_output.append(righthand)
            boxes_output.append(self.rect_roi_coords)
            rects_output.append(rect)

        return (
            landmarks_output,
            handness_output,
            righthands_output,
            boxes_output,
            rects_output,
            world_landmarks_output,
        )

    def get_global_coords(self, landmarks):
        if self.warp_matrix is None:
            new_landmarks = np.zeros_like(landmarks)
            new_landmarks[:, 0] = landmarks[:, 0] + self.rect_roi_coords[0, 0]
            new_landmarks[:, 1] = landmarks[:, 1] + self.rect_roi_coords[0, 1]
        else:
            inv_warp_matrix = np.linalg.inv(self.warp_matrix)
            landmarks2d = cv2.perspectiveTransform(landmarks[None, :, :2], inv_warp_matrix)[0]

            new_landmarks = np.zeros_like(landmarks)
            new_landmarks[:, :2] = landmarks2d.copy()
            new_landmarks[:, 2] = landmarks[:, -1].copy()

        return new_landmarks

    def get_rotated_rect_roi(self, img, rect):
        img_h, img_w = img.shape[0], img.shape[1]
        x_center = int(rect.x_center * img_w)
        y_center = int(rect.y_center * img_h)
        height = int(rect.height * img_h)
        half = 0.5 * height
        rotation_radian = -1.0 * rect.rotation

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

        self.rect_roi_coords = spts.copy()

        return img_roi

    """
    Extracts a subset of the hand landmarks that are relatively more stable across frames
    (e.g. comparing to finger tips) for computing the bounding box. The box will later be expanded to contain the
    entire hand. In this approach, it is more robust to drastically changing hand size.
    The landmarks extracted are: wrist, MCP/PIP of five fingers.
    node {
        calculator: "SplitNormalizedLandmarkListCalculator"
        input_stream: "landmarks"
        output_stream: "partial_landmarks"
        options: {
            [mediapipe.SplitVectorCalculatorOptions.ext] {
                ranges: { begin: 0 end: 4 }
                ranges: { begin: 5 end: 7 }
                ranges: { begin: 9 end: 11 }
                ranges: { begin: 13 end: 15 }
                ranges: { begin: 17 end: 19 }
        combine_outputs: true
            }
        }
    }
    """
    """ A calculator that converts subset of hand landmarks to a bounding box NormalizedRect. The rotation angle of
        the bounding box is computed based on 1) the wrist joint and 2) the average of PIP joints of index finger,
        middle finger and ring finger. After rotation, the vector from the wrist to the mean of PIP joints is expected
        to be vertical with wrist at the bottom and the mean of PIP joints at the top.
        def get_hand_landmarks_to_rect(self, img_size):
    """

    def normalized_landmarks_list_to_rect(self, img_size):
        """
        Mediapipe C++ code:
        absl::Status NormalizedLandmarkListToRect(const NormalizedLandmarkList& landmarks,
        const std::pair<int, int>& image_size, NormalizedRect* rect) {
            const float rotation = ComputeRotation(landmarks, image_size);
            const float reverse_angle = NormalizeRadians(-rotation);

            // Find boundaries of landmarks.
            float max_x = std::numeric_limits<float>::min();
            float max_y = std::numeric_limits<float>::min();
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                max_x = std::max(max_x, landmarks.landmark(i).x());
                max_y = std::max(max_y, landmarks.landmark(i).y());
                min_x = std::min(min_x, landmarks.landmark(i).x());
                min_y = std::min(min_y, landmarks.landmark(i).y());
            }
            const float axis_aligned_center_x = (max_x + min_x) / 2.f;
            const float axis_aligned_center_y = (max_y + min_y) / 2.f;

            // Find boundaries of rotated landmarks.
            max_x = std::numeric_limits<float>::min();
            max_y = std::numeric_limits<float>::min();
            min_x = std::numeric_limits<float>::max();
            min_y = std::numeric_limits<float>::max();
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const float original_x = (landmarks.landmark(i).x() - axis_aligned_center_x) * image_size.first;
                const float original_y = (landmarks.landmark(i).y() - axis_aligned_center_y) * image_size.second;

                const float projected_x = original_x * std::cos(reverse_angle) - original_y * std::sin(reverse_angle);
                const float projected_y = original_x * std::sin(reverse_angle) + original_y * std::cos(reverse_angle);

                max_x = std::max(max_x, projected_x);
                max_y = std::max(max_y, projected_y);
                min_x = std::min(min_x, projected_x);
                min_y = std::min(min_y, projected_y);
            }
            const float projected_center_x = (max_x + min_x) / 2.f;
            const float projected_center_y = (max_y + min_y) / 2.f;

            const float center_x = projected_center_x * std::cos(rotation) - projected_center_y * std::sin(rotation) +
                                   image_size.first * axis_aligned_center_x;
            const float center_y = projected_center_x * std::sin(rotation) + projected_center_y * std::cos(rotation) +
                                   image_size.second * axis_aligned_center_y;
            const float width = (max_x - min_x) / image_size.first;
            const float height = (max_y - min_y) / image_size.second;

            rect->set_x_center(center_x / image_size.first);
            rect->set_y_center(center_y / image_size.second);
            rect->set_width(width);
            rect->set_height(height);
            rect->set_rotation(rotation);

            return absl::OkStatus();
        }
        """
        rotation = self.compute_rotation()
        revese_angle = self.normalize_radians(-rotation)

        # Find boundaries of landmarks.
        max_x = np.max(self.pre_landmarks_subset[:, 0])
        max_y = np.max(self.pre_landmarks_subset[:, 1])
        min_x = np.min(self.pre_landmarks_subset[:, 0])
        min_y = np.min(self.pre_landmarks_subset[:, 1])

        axis_aligned_center_x = (max_x + min_x) * 0.5
        axis_aligned_center_y = (max_y + min_y) * 0.5

        # Find boundaries of rotated landmarks.
        original_x = self.pre_landmarks_subset[:, 0] - axis_aligned_center_x
        original_y = self.pre_landmarks_subset[:, 1] - axis_aligned_center_y

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
        """
        Mediapipe C++ code:
        float ComputeRotation(const NormalizedLandmarkList& landmarks, const std::pair<int, int>& image_size) {
            const float x0 = landmarks.landmark(kWristJoint).x() * image_size.first;
            const float y0 = landmarks.landmark(kWristJoint).y() * image_size.second;

            float x1 = (landmarks.landmark(kIndexFingerPIPJoint).x() +
                        landmarks.landmark(kRingFingerPIPJoint).x()) / 2.f;
            float y1 = (landmarks.landmark(kIndexFingerPIPJoint).y() +
                        landmarks.landmark(kRingFingerPIPJoint).y()) / 2.f;

            x1 = (x1 + landmarks.landmark(kMiddleFingerPIPJoint).x()) / 2.f * image_size.first;
            y1 = (y1 + landmarks.landmark(kMiddleFingerPIPJoint).y()) / 2.f * image_size.second;

            const float rotation = NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
        return rotation;
        }
        """
        self.pre_landmarks_subset = np.zeros((self.num_joints_subset, self.pre_landmarks.shape[1]), dtype=np.float32)
        self.pre_landmarks_subset[0:3] = self.pre_landmarks[0:3].copy()  # Wrist and thumb's two indexes
        self.pre_landmarks_subset[3:5] = self.pre_landmarks[5:7].copy()  # Index MCP & PIP
        self.pre_landmarks_subset[5:7] = self.pre_landmarks[9:11].copy()  # Middle MCP & PIP
        self.pre_landmarks_subset[7:9] = self.pre_landmarks[13:15].copy()  # Ring MCP & PIP
        self.pre_landmarks_subset[9:11] = self.pre_landmarks[17:19].copy()  # Pinky MPC & PIP

        x0, y0 = (
            self.pre_landmarks_subset[self.kWristJoint][0],
            self.pre_landmarks_subset[self.kWristJoint][1],
        )

        x1 = (
            self.pre_landmarks_subset[self.kIndexFingerPIPJoint][0]
            + self.pre_landmarks_subset[self.kRingFingerPIPJoint][0]
        ) * 0.5
        y1 = (
            self.pre_landmarks_subset[self.kIndexFingerPIPJoint][1]
            + self.pre_landmarks_subset[self.kRingFingerPIPJoint][1]
        ) * 0.5
        x1 = (x1 + self.pre_landmarks_subset[self.kMiddleFingerPIPJoint][0]) * 0.5
        y1 = (y1 + self.pre_landmarks_subset[self.kMiddleFingerPIPJoint][1]) * 0.5

        rotation = self.normalize_radians(self.kTargetAngle - math.atan2(-(y1 - y0), x1 - x0))
        return rotation

    """
    Expands the hand rectangle so that the box contains the entire hand and it's big enough so that it's likely to
    still contain the hand even with some motion in the next video frame.
    node {
        calculator: "RectTransformationCalculator"
        input_stream: "NORM_RECT:hand_rect_from_landmarks"
        input_stream: "IMAGE_SIZE:image_size"
        output_stream: "roi"
        options: {
            [mediapipe.RectTransformationCalculatorOptions.ext] {
                scale_x: 2.0
                scale_y: 2.0
                shift_y: -0.1
                square_long: true
            }
        }
    }
    """

    def rect_transformation(self, rect, img_width, img_height):
        """
        Mediapipe C++ Code:
        void RectTransformationCalculator::TransformNormalizedRect(NormalizedRect* rect, int image_width,
        int image_height) {
            float width = rect->width();
            float height = rect->height();
            float rotation = rect->rotation();

            if (options_.has_rotation() || options_.has_rotation_degrees()) {
                rotation = ComputeNewRotation(rotation);
            }
            if (rotation == 0.f) {
                rect->set_x_center(rect->x_center() + width * options_.shift_x());
                rect->set_y_center(rect->y_center() + height * options_.shift_y());
            } else {
                const float x_shift = (image_width * width * options_.shift_x() * std::cos(rotation) -
                    image_height * height * options_.shift_y() * std::sin(rotation)) / image_width;
                const float y_shift = (image_width * width * options_.shift_x() * std::sin(rotation) +
                i   mage_height * height * options_.shift_y() * std::cos(rotation)) / image_height;

            rect->set_x_center(rect->x_center() + x_shift);
            rect->set_y_center(rect->y_center() + y_shift);
            }

            if (options_.square_long()) {
                const float long_side = std::max(width * image_width, height * image_height);
                width = long_side / image_width;
                height = long_side / image_height;
            } else if (options_.square_short()) {
                const float short_side = std::min(width * image_width, height * image_height);
                width = short_side / image_width;
                height = short_side / image_height;
            }

            rect->set_width(width * options_.scale_x());
            rect->set_height(height * options_.scale_y());
        }
        """
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

    @staticmethod
    def normalize_radians(angle):
        """
        Mediapipe C++ code:
        inline float NormalizeRadians(float angle) {
            return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
        }
        """
        return angle - 2 * math.pi * np.floor((angle - (-math.pi)) / (2 * math.pi))

    """
    Projects world landmarks from the rectangle to original coordinates.

    World landmarks are predicted in meters rather than in pixels of the image and have origin in the middle of the
    hips rather than in the corner of the pose image (cropped with given rectangle). Thus only rotation (but not scale
    and translation) is applied to the landmarks to transform them back to original coordinates.

    Input:
        LANDMARKS: A LandmarkList representing world landmarks in the rectangle.
        NORM_RECT: An NormalizedRect representing a normalized rectangle in image coordinates. (Optional)

    Output:
        LANDMARKS: A LandmarkList representing world landmarks projected (rotated but not scaled or translated) from
                    the rectangle to original coordinates.

    Usage example:
    node {
        calculator: "WorldLandmarkProjectionCalculator"
        input_stream: "LANDMARKS:landmarks"
        input_stream: "NORM_RECT:rect"
        output_stream: "LANDMARKS:projected_landmarks"
        }
    """

    @staticmethod
    def world_landmark_projection(landmarks, rect):
        """
        Mediapipe C++ code: refer to
        https://github.com/google/mediapipe/blob/8b57bf879b419173b26277d220b643dac0402334/mediapipe/calculators/
        util/world_landmark_projection_calculator.cc#L91
        """
        new_landmarks = np.zeros_like(landmarks)
        radian = rect.rotation
        new_landmarks[:, 0] = math.cos(radian) * landmarks[:, 0] - math.sin(radian) * landmarks[:, 1]
        new_landmarks[:, 1] = math.sin(radian) * landmarks[:, 0] + math.cos(radian) * landmarks[:, 1]
        new_landmarks[:, 2] = landmarks[:, 2].copy()
        return new_landmarks


class BasicHeatmapHands(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def pre_process(self, img, is_bgr=True):
        res_factor = np.array(
            [img.shape[1] / self.img_size, img.shape[0] / self.img_size],
            dtype=np.float32,
        )
        img_res = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        if is_bgr:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_res

        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        model_input = img_norm[None, ...]
        model_input = np.transpose(model_input, [0, 3, 1, 2])  # NHWC to HCHW

        return model_input, res_factor

    def post_process(self, heatmaps, res_factor):
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        # post-processing
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = heatmaps[n][p]
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

        preds = coords.copy() * (self.img_size / heatmap_height)
        preds *= res_factor

        return preds, maxvals

    def get_img_roi(self, img_bgr, box, scale):
        if isinstance(box, list):  # [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]
            rotation = None
            pass
        elif isinstance(box, dict):
            rotation = box["rotation"]
            box = box["box"]  # {'type': String, 'box': [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]}
        else:
            rotation = None

        ori_img_h = img_bgr.shape[0]
        ori_img_w = img_bgr.shape[1]
        box_w = box[1][0] - box[0][0]
        box_h = box[1][1] - box[0][1]
        new_box = [
            (
                max(0, box[0][0] - 0.5 * (scale - 1.0) * box_w),
                max(0, box[0][1] - 0.5 * (scale - 1.0) * box_h),
            ),
            (
                min(box[1][0] + 0.5 * (scale - 1.0) * box_w, ori_img_w),
                min(box[1][1] + 0.5 * (scale - 1.0) * box_h, ori_img_h),
            ),
        ]

        if rotation is not None:
            img_roi, rect_roi_coord, warp_matrix = self.get_rotated_rect_roi(img_bgr, new_box, rotation=rotation)
            return img_roi, rect_roi_coord, warp_matrix
        else:
            img_roi = img_bgr[
                int(new_box[0][1]) : int(new_box[1][1]),
                int(new_box[0][0]) : int(new_box[1][0]),
            ].copy()
            rect_roi_coord = np.array(
                [
                    [new_box[0][0], new_box[0][1]],  # left-top
                    [new_box[1][0], new_box[0][1]],  # right-top
                    [new_box[0][0], new_box[1][1]],  # left-bottom
                    [new_box[1][0], new_box[1][1]],
                ],
                dtype=np.float32,
            )  # right-bottom
            return img_roi, rect_roi_coord

    def get_rotated_rect_roi(self, img, box, rotation):
        x_center = int(box[0][0] + 0.5 * (box[1][0] - box[0][0]))
        y_center = int(box[0][1] + 0.5 * (box[1][1] - box[0][1]))
        height = int(box[1][1] - box[0][1])

        half = 0.5 * height
        rotation_radian = -1.0 * rotation

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
        warp_matrix = cv2.getPerspectiveTransform(spts, dpts)
        img_roi = cv2.warpPerspective(
            img,
            warp_matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )

        rect_roi_coord = spts.copy()

        return img_roi, rect_roi_coord, warp_matrix

    @staticmethod
    def set_hand_info(hand, landmark, img_roi_bgr, confs, rect_roi_coord=None):
        hand.confs = np.mean(confs)
        if hand.landmark is not None:
            hand.pre_landmark = hand.landmark.copy()

        hand.landmark = landmark.copy()

        if hand.pre_landmark is not None:
            hand.landmark = smooth_pts(hand.pre_landmark, hand.landmark, rect_roi_coord)

        hand.img_roi_bgr = img_roi_bgr.copy()
        if rect_roi_coord is not None:
            hand.rect_roi_coord = rect_roi_coord.copy()
        hand.turn_on()

    @staticmethod
    def get_global_coords(landmark, rect_roi_coord, warp_matrix=None):
        if warp_matrix is None:
            new_landmark = np.zeros_like(landmark)
            new_landmark[:, 0] = landmark[:, 0] + rect_roi_coord[0, 0]
            new_landmark[:, 1] = landmark[:, 1] + rect_roi_coord[0, 1]
        else:
            inv_warp_matrix = np.linalg.inv(warp_matrix)
            landmarks2d = cv2.perspectiveTransform(landmark[None, :, :2], inv_warp_matrix)[0]

            new_landmark = np.zeros_like(landmark)
            new_landmark[:, :2] = landmarks2d.copy()

        return new_landmark

    @staticmethod
    def get_max_preds(heatmaps):
        """
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        """
        assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "heatmaps should be 4-ndim"

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals


class HandLandModelComb(object):
    def __init__(self, capability=1, roi_mode=0, handness_thres=0.5):
        self.roi_mode = roi_mode
        if capability == 0:
            self.model_path = "../lib/models/hand_landmark_lite.onnx"
        elif capability == 1:
            self.model_path = "../lib/models/hand_landmark_full.onnx"
        else:
            raise ValueError(" [!] Capability only supports between 0 and 1!")

        self.img_size = 224
        self.num_joints = 21
        self.dim = 3
        self.handness_thres = handness_thres
        self.righthand_prop_thres = 0.5

        # Load ONXX model
        self.model = onnxruntime.InferenceSession(self.model_path)
        print("*" * 100)
        print(f" [!] HandLandModelRegressor infer-dev: {onnxruntime.get_device()} model: {self.model_path}")

        self.img_size_heatmap = 160
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.model_path_heatmap = "./lib/models/mobilenetv2_160x160_alpha_140_coarserefine_data_v2_4_try03.onnx"
        self.model_heatmap = onnxruntime.InferenceSession(self.model_path_heatmap)

        print("*" * 100)
        print(f" [!] HandLandModelHeatMap infer-dev: {onnxruntime.get_device()} model: {self.model_path_heatmap}")
        print("*" * 100)

    def pre_process(self, img, is_bgr=True):
        res_factor = np.array(
            [img.shape[1] / self.img_size, img.shape[0] / self.img_size],
            dtype=np.float32,
        )
        img_res = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        if is_bgr:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_res

        img_norm = img_rgb.astype(np.float32) / 255.0
        model_input = img_norm[None, ...]
        model_input = np.transpose(model_input, [0, 3, 1, 2])  # NHWC to HCHW

        return model_input, res_factor

    def post_process(self, out, res_factor):
        out = out.reshape(self.num_joints, self.dim)
        out[:, :2] = out[:, :2] * res_factor
        return out

    def run(self, img_bgr, is_get_2d=False, is_bgr=True):
        model_input, res_factor = self.pre_process(img_bgr, is_bgr)

        outputs = self.model.run(None, {self.model.get_inputs()[0].name: model_input})
        landmarks, handness, righthand, world_landmarks = (
            outputs[0][0],
            outputs[1][0],
            outputs[2][0],
            outputs[3][0],
        )
        world_landmarks = world_landmarks.reshape(self.num_joints, self.dim)

        if is_get_2d:
            landmarks = self.post_process(landmarks, res_factor)[:, :2]
        else:
            landmarks = self.post_process(landmarks, res_factor)

        return landmarks, handness, righthand, world_landmarks

    def pre_process_heatmap(self, img, is_bgr=True):
        res_factor = np.array(
            [img.shape[1] / self.img_size_heatmap, img.shape[0] / self.img_size_heatmap],
            dtype=np.float32,
        )
        img_res = cv2.resize(img, (self.img_size_heatmap, self.img_size_heatmap), interpolation=cv2.INTER_CUBIC)

        if is_bgr:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_res

        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        model_input = img_norm[None, ...]
        model_input = np.transpose(model_input, [0, 3, 1, 2])  # NHWC to HCHW

        return model_input, res_factor

    def post_process_heatmap(self, heatmaps, res_factor):
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        # post-processing
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = heatmaps[n][p]
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

        preds = coords.copy() * (self.img_size_heatmap / heatmap_height)
        preds *= res_factor

        preds_3d = np.zeros((1, self.num_joints, 3))
        preds_3d[0, :, :2] = preds.copy()

        return preds_3d, maxvals

    def run_heatmap(self, img_bgr, is_bgr=True):
        model_input, res_factor = self.pre_process_heatmap(img_bgr, is_bgr)
        heatmaps = self.model_heatmap.run(None, {self.model_heatmap.get_inputs()[0].name: model_input})[0]
        landmarks, confs = self.post_process_heatmap(heatmaps, res_factor)
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
                landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
                landmark = hand.get_global_coords(landmark)

                print(f"handness: {handness}")
                if handness >= self.handness_thres:
                    # hand_info and righthand-prop inconsistent
                    if (righthand_prop >= self.righthand_prop_thres and hand.type != "right") or (
                        righthand_prop < self.righthand_prop_thres and hand.type != "left"
                    ):
                        print(" [*] Delete hand-type!")
                        hand.img_roi_bgr = img_roi_bgr.copy()
                        hand.reset()
                    else:
                        self.set_hand_info(
                            hand,
                            landmark,
                            world_landmark,
                            handness,
                            img_roi_bgr,
                            rect_roi_coord=hand.rect_roi_coord,
                        )
                else:
                    print(" [*] Delete handness!")
                    hand.img_roi_bgr = img_roi_bgr.copy()
                    hand.reset()

        for hand in [left_hand, right_hand]:
            if not hand.get_flag():
                hand.reset()

    def handle_hand_detector_bbox(self, img_bgr, left_hand, right_hand, boxes, scale):
        candi_box = list()
        for box in boxes:
            if (right_hand.landmark is not None) and (bb_iou(box, right_hand.landmark_to_box()) > 0.3):
                continue
            elif (left_hand.landmark is not None) and (bb_iou(box, left_hand.landmark_to_box()) > 0.3):
                continue
            else:
                candi_box.append(box)

        for box in candi_box:
            img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=scale)
            landmark, confs = self.run_heatmap(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord)

            is_hand = np.sum(confs > 0.2) >= 11
            if is_hand:  # predicted landmarks are reliable
                if right_hand.landmark is None:
                    self.set_hand_info(
                        hand=right_hand,
                        landmark=landmark,
                        world_landmark=None,
                        handness=["right"],
                        img_roi_bgr=img_roi_bgr,
                        rect_roi_coord=rect_roi_coord,
                    )

                if left_hand.landmark is None:
                    self.set_hand_info(
                        hand=left_hand,
                        landmark=landmark,
                        world_landmark=None,
                        handness=["left"],
                        img_roi_bgr=img_roi_bgr,
                        rect_roi_coord=rect_roi_coord,
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
            img_roi_bgr, rect_roi_coord, warp_matrix = self.get_img_roi(img_bgr, box=box, scale=1)
            landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord, warp_matrix)

            if handness[0] > self.handness_thres:  # predicted landmarks are reliable
                if (
                    (righthand_prop >= self.righthand_prop_thres)
                    and (right_hand.landmark is None)
                    and (box["type"] == "right")
                ):
                    # if (right_hand.landmark is None) and (box["type"] == "right"):
                    self.set_hand_info(
                        right_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
                        rect_roi_coord,
                    )
                if (
                    (righthand_prop < self.righthand_prop_thres)
                    and (left_hand.landmark is None)
                    and (box["type"] == "left")
                ):
                    # if (left_hand.landmark is None) and (box["type"] == "left"):
                    self.set_hand_info(
                        left_hand,
                        landmark,
                        world_landmark,
                        handness,
                        img_roi_bgr,
                        rect_roi_coord,
                    )
            else:
                if box["type"] == "right":
                    right_hand.img_roi_bgr = img_roi_bgr.copy()
                else:
                    left_hand.img_roi_bgr = img_roi_bgr.copy()

        return left_hand, right_hand

    def get_img_roi(self, img_bgr, box, scale):
        if isinstance(box, list):  # [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]
            rotation = None
            pass
        elif isinstance(box, dict):
            rotation = box["rotation"]
            box = box["box"]  # {'type': String, 'box': [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]}
        else:
            rotation = None

        ori_img_h = img_bgr.shape[0]
        ori_img_w = img_bgr.shape[1]
        box_w = box[1][0] - box[0][0]
        box_h = box[1][1] - box[0][1]
        new_box = [
            (
                max(0, box[0][0] - 0.5 * (scale - 1.0) * box_w),
                max(0, box[0][1] - 0.5 * (scale - 1.0) * box_h),
            ),
            (
                min(box[1][0] + 0.5 * (scale - 1.0) * box_w, ori_img_w),
                min(box[1][1] + 0.5 * (scale - 1.0) * box_h, ori_img_h),
            ),
        ]

        if rotation is not None:
            img_roi, rect_roi_coord, warp_matrix = self.get_rotated_rect_roi(img_bgr, new_box, rotation=rotation)
            return img_roi, rect_roi_coord, warp_matrix
        else:
            img_roi = img_bgr[
                int(new_box[0][1]) : int(new_box[1][1]),
                int(new_box[0][0]) : int(new_box[1][0]),
            ].copy()
            rect_roi_coord = np.array(
                [
                    [new_box[0][0], new_box[0][1]],  # left-top
                    [new_box[1][0], new_box[0][1]],  # right-top
                    [new_box[0][0], new_box[1][1]],  # left-bottom
                    [new_box[1][0], new_box[1][1]],
                ],
                dtype=np.float32,
            )  # right-bottom
            return img_roi, rect_roi_coord

    def get_rotated_rect_roi(self, img, box, rotation):
        x_center = int(box[0][0] + 0.5 * (box[1][0] - box[0][0]))
        y_center = int(box[0][1] + 0.5 * (box[1][1] - box[0][1]))
        height = int(box[1][1] - box[0][1])

        half = 0.5 * height
        rotation_radian = -1.0 * rotation

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
        warp_matrix = cv2.getPerspectiveTransform(spts, dpts)
        img_roi = cv2.warpPerspective(
            img,
            warp_matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )

        rect_roi_coord = spts.copy()

        return img_roi, rect_roi_coord, warp_matrix

    @staticmethod
    def set_hand_info(hand, landmark, world_landmark, handness, img_roi_bgr, rect_roi_coord=None):
        if hand.landmark is not None:
            hand.pre_landmark = hand.landmark.copy()
        hand.landmark = landmark.copy()

        # smoothing
        if hand.pre_landmark is not None:
            hand.landmark = smooth_pts(hand.pre_landmark, hand.landmark, rect_roi_coord)

        if world_landmark is not None:
            hand.unprojected_world_landmark = world_landmark.copy()

        hand.handness = handness[0]
        hand.img_roi_bgr = img_roi_bgr.copy()

        if rect_roi_coord is not None:
            hand.rect_roi_coord = rect_roi_coord.copy()

        hand.turn_on()

    @staticmethod
    def get_global_coords(landmark, rect_roi_coord, warp_matrix=None):
        if warp_matrix is None:
            new_landmark = np.zeros_like(landmark)
            new_landmark[:, 0] = landmark[:, 0] + rect_roi_coord[0, 0]
            new_landmark[:, 1] = landmark[:, 1] + rect_roi_coord[0, 1]
        else:
            inv_warp_matrix = np.linalg.inv(warp_matrix)
            landmarks2d = cv2.perspectiveTransform(landmark[None, :, :2], inv_warp_matrix)[0]

            new_landmark = np.zeros_like(landmark)
            new_landmark[:, :2] = landmarks2d.copy()
            new_landmark[:, 2] = landmark[:, -1].copy()

        return new_landmark

    @staticmethod
    def get_max_preds(heatmaps):
        """
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        """
        assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "heatmaps should be 4-ndim"

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals


class HandLandModelCombTracker(HandLandModelComb):
    def __init__(self, capability=1, roi_mode=0, handness_thres=0.5):
        super(HandLandModelCombTracker, self).__init__(
            capability=capability, roi_mode=roi_mode, handness_thres=handness_thres
        )

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
                landmark, handness, righthand_prop, world_landmark = self.run(img_roi_bgr, is_bgr=True)
                landmark = hand.get_global_coords(landmark)

                print(f"handness: {handness}")
                if handness >= self.handness_thres:
                    # hand_info and righthand-prop inconsistent
                    if (righthand_prop >= self.righthand_prop_thres and hand.type != "right") or (
                        righthand_prop < self.righthand_prop_thres and hand.type != "left"
                    ):
                        print(" [*] Delete hand-type!")
                        hand.img_roi_bgr = img_roi_bgr.copy()
                        hand.reset()
                        # hand.release_kalman()
                    else:
                        self.set_hand_info(
                            hand,
                            landmark,
                            world_landmark,
                            handness,
                            img_roi_bgr,
                            rect_roi_coord=hand.rect_roi_coord,
                        )
                else:
                    print(" [*] Delete handness!")
                    hand.img_roi_bgr = img_roi_bgr.copy()
                    hand.reset()

        for hand in [left_hand, right_hand]:
            if not hand.get_flag():
                hand.reset()

            # update tracker
            hand.track_box = self.update_kalman(hand)  # update kalman-tracker

    def handle_hand_detector_bbox(self, img_bgr, left_hand, right_hand, boxes, scale):
        candi_box = list()
        for box in boxes:
            if (right_hand.landmark is not None) and (bb_iou(box, right_hand.landmark_to_box()) > 0.3):
                continue
            elif (left_hand.landmark is not None) and (bb_iou(box, left_hand.landmark_to_box()) > 0.3):
                continue
            else:
                candi_box.append(box)
        # num_dets = len(candi_box)
        # print(f'num_dets: {num_dets}')

        for hand in [left_hand, right_hand]:
            if hand.landmark is None:
                for box in hand.track_box:
                    candi_box.append([(box[0], box[1]), (box[2], box[3])])
        # num_tracks = len(candi_box) - num_dets
        # print(f'num_tracks: {num_tracks}')

        for box in candi_box:
            img_roi_bgr, rect_roi_coord = self.get_img_roi(img_bgr, box=box, scale=scale)
            landmark, confs = self.run_heatmap(img_roi_bgr, is_bgr=True)
            landmark = self.get_global_coords(landmark, rect_roi_coord)

            is_hand = np.sum(confs > 0.2) >= 12
            if is_hand:  # predicted landmarks are reliable
                # print(f'np.sum(confs > 0.2): {np.sum(confs > 0.2)}')
                if right_hand.landmark is None:
                    self.set_hand_info(
                        hand=right_hand,
                        landmark=landmark,
                        world_landmark=None,
                        handness=["right"],
                        img_roi_bgr=img_roi_bgr,
                        rect_roi_coord=rect_roi_coord,
                    )

                if left_hand.landmark is None:
                    self.set_hand_info(
                        hand=left_hand,
                        landmark=landmark,
                        world_landmark=None,
                        handness=["left"],
                        img_roi_bgr=img_roi_bgr,
                        rect_roi_coord=rect_roi_coord,
                    )

        return left_hand, right_hand

    @staticmethod
    def update_kalman(hand):
        if hand.landmark is None:
            box_arr = np.array([])
            label_arr = np.array([])
        else:
            box = hand.landmark_to_box()
            box_arr = np.array([[box[0][0], box[0][1], box[1][0], box[1][1]]])
            label_arr = np.zeros(box_arr.shape[0])

        new_box = hand.kalman.update(box_arr, label_arr)
        return new_box
