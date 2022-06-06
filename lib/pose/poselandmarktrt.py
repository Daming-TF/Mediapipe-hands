import math
import platform
import numpy as np

from lib.core_trt.api.human_action_config import config
from lib.core_trt.api.human_action_detector import HumanActionDetector


class PoseLandmarkTRT(object):
    def __init__(self):
        self.num_humans = 1
        self.num_joints = 25
        self.num_dims = 3
        self.LEFT_SHOULDER_INDEX = 5
        self.RIGHT_SHOULDER_INDEX = 6
        self.LEFT_WRIST_INDEX = 9
        self.RIGHT_WRIST_INDEX = 10
        self.LEFT_HAND_INDEX = 17
        self.RIGHT_HAND_INDEX = 18
        self.thres = 0.5
        self.landmark = None
        self.kTargetAngle = math.pi * 0.5  # 90 degree represented in radian

        sysstr = platform.system()
        if sysstr == "Linux":
            lib_name = "./lib/models/pose_trt/libhuya_face.so"
        else:
            raise Exception(" [!] TRT Pose Landmark Only support Ubuntu environment!")

        self.detector = HumanActionDetector(lib_name, config.vidCreateConfig)
        self.detector.addSubModel(b"./lib/models/pose_trt/hyai_pc_sdk_wholebody_v1.6.0_trt.model")
        self.detector.setParam(config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACELIMIT, 8)

    def __call__(self, img_bgr):
        pred_result = self.detector.run(
            img_bgr,
            config.PixelFormat.HYPixelFormatBGR,
            config.ImageRotate.ImageRotationCCW0,
            config.detectConfig,
        )
        hand_boxes = self.post_process(pred_result, img_bgr.shape)
        self.convert_to_array(pred_result)
        return hand_boxes, self.landmark

    def convert_to_array(self, det_result):
        if det_result.human_count > 0:
            self.landmark = np.zeros((self.num_humans, self.num_joints, self.num_dims), dtype=np.float32)
            for j in range(self.num_humans):
                for land_id in range(self.num_joints):
                    self.landmark[j, land_id, :] = np.array(
                        [
                            det_result.d_humans[j].points_array[land_id].x,
                            det_result.d_humans[j].points_array[land_id].y,
                            det_result.d_humans[j].keypoints_score[land_id],
                        ],
                        dtype=np.float32,
                    )
        else:
            self.landmark = None

    def post_process(self, det_result, img_shape):
        boxes = list()

        if det_result.human_count > 0:
            for j in range(self.num_humans):
                left_shoulder = np.array(
                    [
                        det_result.d_humans[j].points_array[self.LEFT_SHOULDER_INDEX].x,
                        det_result.d_humans[j].points_array[self.LEFT_SHOULDER_INDEX].y,
                    ]
                )
                right_shoulder = np.array(
                    [
                        det_result.d_humans[j].points_array[self.RIGHT_SHOULDER_INDEX].x,
                        det_result.d_humans[j].points_array[self.RIGHT_SHOULDER_INDEX].y,
                    ]
                )
                shoulder_len = np.sqrt(np.sum(np.power((left_shoulder - right_shoulder), 2)))

                for i, (hand_index, wrist_index) in enumerate(
                    zip(
                        [self.LEFT_HAND_INDEX, self.RIGHT_HAND_INDEX],
                        [self.LEFT_WRIST_INDEX, self.RIGHT_WRIST_INDEX],
                    )
                ):

                    if det_result.d_humans[j].keypoints_score[hand_index] > self.thres:
                        wrist = np.array(
                            [
                                det_result.d_humans[j].points_array[wrist_index].x,
                                det_result.d_humans[j].points_array[wrist_index].y,
                            ]
                        )

                        # out of boundary
                        if (wrist[0] > img_shape[1]) or (wrist[1] > img_shape[0]):
                            break

                        hand = np.array(
                            [
                                det_result.d_humans[j].points_array[hand_index].x,
                                det_result.d_humans[j].points_array[hand_index].y,
                            ]
                        )
                        hand_len = np.sqrt(np.sum(np.power((wrist - hand), 2)))
                        wh = np.maximum(hand_len * 1.4, shoulder_len * 0.5)

                        box = self.get_handbbox(hand, wh, img_shape)
                        rotation = self.normalized_landmarks_list_to_rect(np.vstack([wrist, hand]))

                        if (i == 0) and (box is not None):
                            boxes.append({"type": "left", "box": box, "rotation": rotation})
                        elif (i == 1) and (box is not None):
                            boxes.append({"type": "right", "box": box, "rotation": rotation})

        return boxes

    def normalized_landmarks_list_to_rect(
        self,
        landmark,
    ):
        rotation = self.compute_rotation(landmark)
        return rotation

    def compute_rotation(self, landmark):
        x0 = landmark[0, 0]
        y0 = landmark[0, 1]
        x1 = landmark[1, 0]
        y1 = landmark[1, 1]

        rotation = self.normalize_radians(self.kTargetAngle - math.atan2(-(y1 - y0), x1 - x0))

        return rotation

    @staticmethod
    def normalize_radians(angle):
        return angle - 2 * math.pi * np.floor((angle - (-math.pi)) / (2 * math.pi))

    @staticmethod
    def get_handbbox(hand, wh, img_shape):
        h, w = img_shape[0], img_shape[1]

        bbox = np.array(
            [[hand[0] - wh, hand[1] - wh], [hand[0] + wh, hand[1] + wh]],
            dtype=np.float32,
        )
        bbox[0][0] = np.maximum(0, bbox[0][0])
        bbox[0][1] = np.maximum(0, bbox[0][1])
        bbox[1][0] = np.minimum(bbox[1][0], w - 1)
        bbox[1][1] = np.minimum(bbox[1][1], h - 1)

        if (int(bbox[1][0]) - int(bbox[0][0]) > 0) and (int(bbox[1][1]) - int(bbox[0][1]) > 0):
            return bbox  # box size should be bigger than 0
        else:
            return None

    def release(self):
        self.detector.reset()
        self.detector.destroy()
