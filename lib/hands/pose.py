import cv2
import os
import platform
import numpy as np

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if platform.system() == "Windows":
    os.environ["path"] += ";" + os.path.join(current_dir, "models/pose")

from lib.core.api.human_action_config import config
from lib.core.api.human_action_detector import HumanActionDetector


class PoseLandmark(object):
    def __init__(self):
        # 请注意 cpu模式和gpu模式以及不同平台分别加载不同的动态库以及不同的模型文件
        sysstr = platform.system()
        if sysstr == "Linux":
            lib_name = "libhuya_face.so"
        elif sysstr == "Windows":
            lib_name = "huya_face.dll"
        else:
            raise Exception(" [!] Pose Landmark only support Linux and Windows")

        self.detector = HumanActionDetector(lib_name, config.vidCreateConfig)
        self.detector.addSubModel(b"./lib/models/pose/hyai_pc_sdk_body_v1.0.2.model")
        self.detector.setParam(config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACELIMIT, 8)

    def __call__(self, img_bgr):
        pred_result = self.detector.run(
            img_bgr,
            config.PixelFormat.HYPixelFormatBGR,
            config.ImageRotate.ImageRotationCCW0,
            config.detectConfig,
        )

        if pred_result.human_count > 0:
            for j in range(1):  # for j in range(pred_result.human_count):
                for land_id in range(17, 18, 1):
                    if pred_result.d_humans[j].keypoints_score[land_id] > 0.5:
                        x = pred_result.d_humans[j].points_array[land_id].x
                        y = pred_result.d_humans[j].points_array[land_id].y

        return pred_result
