import cv2
import os
import sys
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.utils.video import VideoWriter
from lib.evals.hands_trackerv3 import HandsTrackerv3BaseAlign


parser = ArgumentParser()
parser.add_argument("--camera", action="store_true", help="open camera, default is test video")
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode show bbox and some other information",
)
parser.add_argument(
    "--roi_mode",
    type=int,
    default=0,
    choices=[0, 1],
    help="0: hand-detector, 1: pose-landmark for providing the position of hands",
)
parser.add_argument("--inp", default=r"./videos/hand_test_02.mp4")
parser.add_argument(
    "--fps",
    default=30,
    type=int,
    help="frame-per-second used in saving video when works on webcam mode",
)
parser.add_argument("--save_path", default=r"./saves/evalv3_base_align_camera_test.mp4")

opts = parser.parse_args()


def main():
    # For webcam input:
    if opts.camera:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, opts.fps)
    else:
        cap = cv2.VideoCapture(opts.inp)

    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    tracker = HandsTrackerv3BaseAlign(roi_mode=opts.roi_mode, debug=opts.debug)
    writer = VideoWriter(
        opts.save_path
        if opts.camera
        else os.path.join(os.path.dirname(opts.save_path), "evalv3-base-align-" + os.path.basename(opts.inp)),
        cap,
        (w, h),
    )

    counter = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")
            break

        if opts.camera:
            frame = cv2.flip(frame, flipCode=1)
        canvas = tracker(frame, counter)
        cv2.imshow("Window", canvas)
        writer.write(canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            cap.release()
            break

        counter += 1

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
