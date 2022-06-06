import cv2
import os
import sys
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.utils.video import VideoWriter
from lib.hands.hand_tracker import HandTracker


parser = ArgumentParser()
parser.add_argument("--camera", action="store_true", help="open camera, default is test video")
parser.add_argument("--draw3d", action="store_true", help="draw world landmarks in 3d space")

parser.add_argument(
    "--roi_mode",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="0: hand-detector; 1: pose-landmark; 2: pre-defined roi",
)
parser.add_argument(
    "--capability",
    default=1,
    type=int,
    choices=[0, 1],
    help="model capability, 1 for large and 0 for lite model",
)
parser.add_argument(
    "--pipe_mode",
    default=1,
    type=int,
    choices=[0, 1],
    help="pipepline mode, 0 indicates our pipeline, 1 is the mediapipe's rotated pipline mode",
)
parser.add_argument("--inp", default=r"./videos/hand_test_02.mp4")
parser.add_argument(
    "--fps",
    default=30,
    type=int,
    help="frame-per-second used in saving video when works on webcam mode",
)
parser.add_argument("--save_path", default=r"./saves/test.mp4")
opts = parser.parse_args()


def main():
    # For webcam input:
    if opts.camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, opts.fps)
    else:
        cap = cv2.VideoCapture(opts.inp)

    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    # capability 0 for the light model and 1 for large model
    tracker = HandTracker(
        frame_size=(h, w),
        capability=opts.capability,
        pipe_mode=opts.pipe_mode,
        is_draw3d=opts.draw3d,
        roi_mode=opts.roi_mode,
    )
    writer = VideoWriter(opts.save_path, cap, (2 * w, h) if opts.draw3d else (w, h))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
            break

        frame = cv2.flip(frame, flipCode=1)  # Flip the image horizontally for a later selfie-view display.
        canvas = tracker(frame)
        cv2.imshow("Window", canvas)
        writer.write(canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            cap.release()
            break

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
