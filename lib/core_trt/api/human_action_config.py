# coding=UTF-8
"""
@Date: 2020-05-25 15:39:10
@LastEditors: zhiwen
@Description: file content
"""

import os
from easydict import EasyDict as edict

config = edict()

#    0          1            2            3
#
#  888888   8888888888        88      88
#  88           88  88        88      88 88  88
#  8888             88      8888      8888888888
#  88                         88
#  88                     888888

# 旋转角度，顺时针方向
config.ImageRotate = edict()
config.ImageRotate.ImageRotationCCW0 = 0
config.ImageRotate.ImageRotationCCW90 = 1
config.ImageRotate.ImageRotationCCW180 = 2
config.ImageRotate.ImageRotationCCW270 = 3

# 图像格式
config.PixelFormat = edict()
config.PixelFormat.HYPixelFormatRGBA = 0
config.PixelFormat.HYPixelFormatBGRA = 1
config.PixelFormat.HYPixelFormatYUVNV21 = 2
config.PixelFormat.HYPixelFormatYUVNV12 = 3  # Y+UV
config.PixelFormat.HYPixelFormatRGB = 5
config.PixelFormat.HYPixelFormatBGR = 7
config.PixelFormat.HYPixelFormatYUV420P = 8

config.ResultCode = edict()  # SDK返回的状态码
config.ResultCode.HY_OK = 0  # 正常运行
config.ResultCode.HY_E_INVALIDARG = -1  # 无效参数
config.ResultCode.HY_E_HANDLE = -2  # 句柄错误
config.ResultCode.HY_E_FAIL = -4  # 内部错误
config.ResultCode.HY_E_INVALID_PIXEL_FORMAT = -6  # 不支持的图像格式
config.ResultCode.HY_E_FILE_NOT_FOUND = -7  # 文件不存在
config.ResultCode.HY_E_INVALID_FILE_FORMAT = -8  # 文件格式不正确导致加载失败
config.ResultCode.HY_E_FACE_DETECTOR_INIT = -10  # 人脸检测器初始化失败
config.ResultCode.HY_E_STATE_CONFILICT = -11  # 检测器的检测和重置或者destroy状态冲突
config.ResultCode.HY_E_DETECT_CONFIG_PARAM = -12  # 检测配置参数错误

config.SDKParamType = edict()
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACELIMIT = 1  # 设置最大人脸个数，默认值是5
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACEACTION_EYE_THRESHOLD = 2  # 设置左闭眼或者右闭眼的阈值 范围[0， 100]，值越小越灵敏， 默认值50
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACEACTION_MOUTH_THRESHOLD = 3  # 设置嘴巴张开的阈值 范围[0， 100]，值越小越灵敏， 默认值50
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACEACTION_HEAD_THRESHOLD = 4  # 设置左右摇头的阈值 范围[0， 100]，值越小越灵敏， 默认值50
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACE_DETECT_SMOOTH_THRESHOLD = (
    5  # 设置人脸106点的阈值[0.0, 1.0], pc端默认0.5，移动端默认值1.0，值越大, 点越稳定, 但相应点会有滞后. 例如0.0是无平滑，0.5轻度平滑，1.0重度平滑。
)
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACE_EXTRA_SMOOTH_THRESHOLD = (
    6  # 设置人脸134点的阈值[0.0, 1.0], pc端默认0.5，移动端默认值1.0，值越大, 点越稳定, 但相应点会有滞后. 例如0.0是无平滑，0.5轻度平滑，1.0重度平滑。
)
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACE_DETECT_INTERVAL = (
    7  # 设置tracker每多少帧进行一次detect(默认值有人脸时30,无人脸时每一帧都检测).设置范围是[5, 50] 值越大,cpu占用率越低, 但检测出新人脸的时间越长.
)
config.SDKParamType.HYAI_MOBILE_SDK_PARAM_FACE_DETECT_INITNUM = (
    8  # 设置连续有多少帧人脸的时候才返回结果，默认是2帧. 设置范围是[0, 3], 值越小从无到有返回人脸的速度越快
)


# 人脸动作 检测的时候使用
config.HY_MOBILE_FACE_DETECT = 0x00000001
config.HY_MOBILE_DETECT_SRC_FACE_POINTS = 0x00000010
config.HY_MOBILE_DETECT_EXTRA_FACE_POINTS = 0x01000000  # 人脸240关键点
config.HY_MOBILE_FACE_LMK_ONLY = 0x00000004  # 只检测关键点，需要自己输入框
config.HY_MOBILE_BODY_KEYPOINTS = 0x08000000
config.HY_MOBILE_DETECT_FACE_GAZE = 0x00000040

# 创建人体行为检测句柄配置选项
# 支持的检测类型 (创建句柄的时候使用)
config.HY_MOBILE_ENABLE_FACE_DETECT = 0x00000040  # 检测人脸
config.HY_MOBILE_ENABLE_FACE_EXTRA_DETECT = 0x00000200  # 检测人脸240
config.HY_MOBILE_ENABLE_BODY_KEYPOINTS = 0x00001000
config.HY_MOBILE_ENABLE_DEBUG_LOG = 0x10000000  # 是否打印调试的log信息
config.HY_MOBILE_ENABLE_GAZE = 0x00000004

# 检测模式
config.HY_MOBILE_DETECT_MODE_VIDEO = 0x00020000  # 视频检测
config.HY_MOBILE_DETECT_MODE_IMAGE = 0x00040000  # 图片检测

config.HY_MOBILE_BACKEND_CPU = 0x00000001  # GPU模式
config.HY_MOBILE_BACKEND_GPU = 0x00000002  # CPU模式

# 图片模式的createConfig
config.picCreateConfig = config.HY_MOBILE_DETECT_MODE_IMAGE | config.HY_MOBILE_BACKEND_CPU
# 视频模式的createConfig
config.vidCreateConfig = config.HY_MOBILE_DETECT_MODE_VIDEO | config.HY_MOBILE_BACKEND_CPU
## GPU模式
# config.vidGpuCreateConfig = config.HY_MOBILE_ENABLE_FACE_DETECT | config.HY_MOBILE_ENABLE_FACE_EXTRA_DETECT | config.HY_MOBILE_DETECT_MODE_VIDEO | config.HY_MOBILE_BACKEND_GPU

config.detectConfig = (
    config.HY_MOBILE_FACE_DETECT
    | config.HY_MOBILE_DETECT_EXTRA_FACE_POINTS
    | config.HY_MOBILE_DETECT_SRC_FACE_POINTS
    | config.HY_MOBILE_DETECT_FACE_GAZE
    | config.HY_MOBILE_BODY_KEYPOINTS
)
config.detectLMKOnlyConf = config.HY_MOBILE_FACE_DETECT | config.HY_MOBILE_FACE_LMK_ONLY
