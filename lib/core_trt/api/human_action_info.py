# coding=UTF-8
"""
@Date: 2020-05-25 15:59:25
@LastEditors: zhiwen
@Description: file content
"""
from ctypes import (
    c_float,
    c_int,
    Structure,
    POINTER,
    c_void_p,
    c_ubyte,
    c_int64,
    c_ulonglong,
)


class hy_pointf_t(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class HYRect(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class hy_mobile_106_t(Structure):
    _fields_ = [
        ("rect", HYRect),
        ("score", c_float),
        ("points_array", hy_pointf_t * 106),
        ("visibility_array", c_float * 106),
        ("yaw", c_float),
        ("pitch", c_float),
        ("roll", c_float),
    ]


class blendshape_info_t(Structure):
    _fields_ = [
        ("blendShapeLocation", c_float * 53),
        ("blendShapePose", c_float * 3),
        ("blendShapeMatrix", c_float * 16),
    ]


class HYFaceInfo(Structure):
    _fields_ = [
        ("face106", hy_mobile_106_t),
        ("p_extra_face_points", hy_pointf_t * 134),
        ("extra_face_points_count", c_int),
        ("p_src_face_points", hy_pointf_t * 223),
        ("src_face_points_count", c_int),
        ("p_left_eyeball_points", hy_pointf_t * 20),
        ("left_eyeball_points_count", c_int),
        ("p_right_eyeball_points", hy_pointf_t * 20),
        ("right_eyeball_points_count", c_int),
        ("p_forhead_points", hy_pointf_t * 36),
        ("forhead_points_count", c_int),
        ("left_gaze_direction", c_float * 3),
        ("left_gaze_score", c_float),
        ("right_gaze_direction", c_float * 3),
        ("right_gaze_score", c_float),
        ("tongue_state", c_float * 3),
        ("bs_info", blendshape_info_t),
        ("face_action", c_ulonglong),
    ]


class HY3DFaceInfo(Structure):
    _fields_ = [
        ("triangleCount", c_int),
        ("triangleIndices", c_void_p),
        ("vertices", c_void_p),
        ("textureCoordinates", c_void_p),
        ("pointsCount", c_int),
        ("affineMatrix", c_float * 16),
        ("projM", c_float * 16),
        ("viewM", c_float * 16),
    ]


class HYHumanPoseInfo(Structure):
    _fields_ = [
        ("rect", HYRect),
        ("score", c_float),
        ("points_array", hy_pointf_t * 25),
        ("keypoints_score", c_float * 25),
        ("key_points_count", c_int),
        ("pose_action", c_int),
    ]


class HYHumanActions(Structure):
    _fields_ = [
        ("d_faces", POINTER(HYFaceInfo)),
        ("faces_count", c_int),
        ("d_humans", POINTER(HYHumanPoseInfo)),
        ("human_count", c_int),
        ("d_faces_3d", POINTER(HY3DFaceInfo)),
        ("faces_3d_count", c_int),
    ]
