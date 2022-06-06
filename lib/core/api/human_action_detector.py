# coding=UTF-8
"""
@Date: 2020-05-25 14:19:10
@LastEditors: zhiwen
@Description: file content
"""

from ctypes import *
import numpy as np
from lib.core.api.human_action_info import HYHumanActions
from lib.core.api.human_action_config import config


class HumanActionDetector:
    """
    huya face tracking python api

    初始化函数
    Parameters:
        shared_lib_path: 动态库的路径
        model_path: 模型文件的路径
        createConfig: 创建检测器的配置参数，注意选择视频模式和图片模式，针对不同应用场景，具体参考config.picCreateConfig或者config.vidCreateConfig
    Returns:

    """

    def __init__(self, shared_lib_path, createConfig=config.vidCreateConfig):
        self.lib = CDLL(shared_lib_path, RTLD_GLOBAL)
        initDetector = self.lib.hyInitModel
        initDetector.argtypes = [c_int]
        initDetector.restype = c_void_p
        self.net = initDetector(createConfig)

    """
    添加模型接口
    Parameters:
        model_path：模型文件路径
    Returns:
        errCode: 成功调用返回 0
    """

    def addSubModel(self, model_path):
        addModelFunc = self.lib.hyAddSubModelFromPath
        addModelFunc.argtypes = [c_void_p, c_char_p]
        addModelFunc.restype = c_int

        errCode = addModelFunc(self.net, model_path)

        return errCode

    """
    设置参数接口
    Parameters:
        type: 要设置的参数类型，参考config.SDKParamType
        value: 对应要设置的参数的值
    Returns:
        errCode 返回状态码，参考config.ResultCode
    """

    def setParam(self, type, value):
        setParamFunc = self.lib.hySetModelParam

        setParamFunc.argtypes = [c_void_p, c_int, c_float]
        setParamFunc.restype = c_int

        errCode = setParamFunc(self.net, type, value)

        return errCode

    """
    检测函数
    Parameters:
        img: 需要检测的数据，[h, w, c]格式排列
        pixFormat: 检测图像的数据格式，目前支持RGB、BGR、RGBA、BRGA、NV21、NV12, 具体参考config.PixelFormat
        imgRotate: 检测图像的旋转方向，具体参考config.ImageRotate
        detectConfig: 检测的配置选项，可选配置检测106点，检测134点，检测肢体关键点，具体使用方法参考config.detectConfig
    Returns:
        detResult: 检测到的相关信息，参考face_info
    """

    def run(self, img, pixFormat, imgRotate, detectConfig):

        imgH, imgW, _ = img.shape
        widthStep = img.strides[0]
        img = img.astype(np.uint8)
        data = img.ctypes.data_as(POINTER(c_ubyte))

        runDetector = self.lib.hyExecuteDetect
        runDetector.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int64,
            POINTER(HYHumanActions),
        ]
        runDetector.restype = c_int

        detResult = HYHumanActions()
        errCode = runDetector(
            self.net,
            data,
            pixFormat,
            imgW,
            imgH,
            widthStep,
            imgRotate,
            detectConfig,
            detResult,
        )

        return detResult

    """
    只检测关键点函数
    Parameters:
        img: 需要检测的数据，[h, w, c]格式排列
        pixFormat: 检测图像的数据格式，目前支持RGB、BGR、RGBA、BRGA、NV21、NV12, 具体参考config.PixelFormat
        imgRotate: 检测图像的旋转方向，具体参考config.ImageRotate
        detectConfig: 检测的配置选项，可选配置检测106点，检测134点，检测肢体关键点，具体使用方法参考config.detectConfig
    Returns:
        detResult: 检测到的相关信息，参考face_info
    """

    def det_lmk_only(self, img, pixFormat, imgRotate, bbox, detectConfig):
        img = img.astype(np.uint8)
        imgH, imgW, _ = img.shape
        widthStep = img.strides[0]
        data = img.ctypes.data_as(POINTER(c_ubyte))

        bbox_left = int(bbox[0])
        bbox_top = int(bbox[1])
        bbox_width = int(bbox[2])
        bbox_height = int(bbox[3])

        runLmkDetector = self.lib.hyExecuteLmkOnly
        runLmkDetector.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int64,
            c_int,
            c_int,
            c_int,
            c_int,
            POINTER(HYHumanActions),
        ]
        runLmkDetector.restype = c_int

        detResult = HYHumanActions()
        errCode = runLmkDetector(
            self.net,
            data,
            pixFormat,
            imgW,
            imgH,
            widthStep,
            imgRotate,
            detectConfig,
            bbox_left,
            bbox_top,
            bbox_width,
            bbox_height,
            detResult,
        )

        return detResult

    """
    重置函数，重置底层状态信息，通常在新的视频流进入之前调用
    Parameters:
    Returns:
        errCode: 成功调用返回 0
    """

    def reset(self):
        resetDetector = self.lib.hyResetHandle
        resetDetector.argtypes = [c_void_p]
        resetDetector.restype = c_int

        errCode = resetDetector(self.net)

        return errCode

    """
    销毁句柄，释放底层内存，在不需要检测器的时候调用
    Parameters:
    Returns:
        errCode: 成功调用返回 0
    """

    def destroy(self):
        destroyDetector = self.lib.hyDestroyHandle
        destroyDetector.argtypes = [c_void_p]
        destroyDetector.restype = c_int

        errCode = destroyDetector(self.net)

        return errCode
