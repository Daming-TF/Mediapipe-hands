import time
import numpy as np

from lib.hands.hands import Hands, MediapipeHands
from lib.hands.detector import HandDetModel
from lib.hands.pose import PoseLandmark
from lib.utils.draw import (
    draw_point,
    draw_rectangle,
    draw_rotated_rect,
    draw_text,
    copy_past_roi,
    Draw3dLandmarks,
    draw_gesture,
)
from lib.utils.gesture import recognize_gesture
from lib.utils.utils import smooth_pts, coord_to_box


class HandTracker(object):
    def __init__(
        self,
        frame_size=None,
        capability=1,
        threshold=0.5,
        pipe_mode=0,
        is_draw3d=False,
        roi_mode=0,
    ):
        if frame_size is None:
            self.priori_box = [
                (300, 200),
                (700, 500),
            ]  # get a priori box with kinect or hand detector
        else:
            h, w = frame_size[0], frame_size[1]
            self.priori_box = [(0.25 * w, 0.25 * h), (0.75 * w, 0.75 * h)]

        self.hand_boxes = None
        self.pts_buffer = [None]
        self.landmark_thres = threshold
        self.pipe_mode = pipe_mode
        self.roi_mode = roi_mode

        self._init_models(is_draw3d, frame_size, capability)

    def _init_models(self, is_draw3d, frame_size, capability):
        if is_draw3d:
            self.draw3der = Draw3dLandmarks(frame_size)
        else:
            self.draw3der = None

        if self.roi_mode == 0:
            self.detector = HandDetModel()  # hand detector
        elif self.roi_mode == 1:
            self.detector = PoseLandmark()  # pose landmark
        else:
            pass  # self.roi_mode == 2, pre-defined roi

        self.name = "TFLite-Full" if capability > 0 else "TFLite-Lite"
        if self.pipe_mode == 0:
            self.hand_model = Hands(capability)  # using our original pipeline logic
        else:
            self.hand_model = MediapipeHands(capability)  # using mediapipe's rotated rectangled roi logic

    def __call__(self, img_bgr):
        img_show = img_bgr.copy()

        if (self.roi_mode != 2) and (self.hand_boxes is None):
            priori_box = self.detector(img_bgr)
            if len(priori_box) > 0:
                self.hand_boxes = priori_box.copy()
            else:
                self.hand_boxes = []
        elif self.hand_boxes is None:
            self.hand_boxes = [
                self.priori_box.copy(),
            ]

        start = time.time()
        (
            pose_preds,
            handness,
            righthand_props,
            roi_boxes,
            rects,
            world_landmarks,
        ) = self.hand_model.run_with_boxes(img_bgr, self.hand_boxes)
        end = time.time()
        print(f"Landmark time: {(end - start) * 1000:.2f} ms. - {self.name}")

        hand_boxes_tmp = []
        pts_bufffer_tmp = []
        for (coords, is_hand, righthand_prop, coords_last, hand_box, roi_box, rect, world_landmark,) in zip(
            pose_preds,
            handness,
            righthand_props,
            self.pts_buffer,
            self.hand_boxes,
            roi_boxes,
            rects,
            world_landmarks,
        ):

            if is_hand > self.landmark_thres:
                if coords_last is not None:
                    coords = smooth_pts(coords_last, coords, hand_box)

                box = coord_to_box(coords)
                hand_boxes_tmp.append(box)
                pts_bufffer_tmp.append(coords)

                img_show = draw_text(img_show, is_hand, righthand_prop, rect)
                img_show = draw_point(img_show, coords)
                img_show = copy_past_roi(img_show, self.hand_model.img_roi_bgr)

                if self.draw3der is not None:  # concat world-landmarks on right side of the img
                    img_show = self.draw3der(img_show, world_landmark)

                if self.pipe_mode == 0:
                    img_show = draw_rectangle(img_show, roi_box)
                else:  # mediapipe's rotated roi pipeline
                    img_show = draw_rotated_rect(img_show, self.hand_model.rect_roi_coords)
                    # draw gesture label
                    img_show = draw_gesture(
                        img_show,
                        coords,
                        recognize_gesture(self.hand_model.unprojected_world_landmarks),
                    )
            else:
                if self.draw3der is not None:
                    pad_img = 255 * np.ones((img_bgr.shape[0], img_bgr.shape[0], 3), dtype=np.uint8)
                    img_show = np.hstack([img_show, pad_img])
                if self.roi_mode == 2:
                    img_show = draw_rectangle(img_show, self.priori_box)  # draw initial roi_box

        self.hand_boxes = hand_boxes_tmp.copy()
        self.pts_buffer = pts_bufffer_tmp.copy()

        if len(self.hand_boxes) == 0 or len(self.pts_buffer) == 0:
            # self.hand_boxes = [self.priori_box.copy(), ]
            self.hand_boxes = None
            self.pts_buffer = [
                None,
            ]
            self.hand_model.clear_history()

        return img_show
