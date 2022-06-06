import cv2
import os
import sys
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.utils.video import VideoWriter
from lib.hands.hands_tracker import HandsVisualTracker


parser = ArgumentParser()
parser.add_argument("--camera", action="store_true", help="open camera, default is test video")
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode show bbox and some other information",
)

parser.add_argument(
    "--capability",
    default=0,
    type=int,
    choices=[0, 1],
    help="model capability, 1 for large and 0 for lite hand-models",
)
parser.add_argument("--inp", default=r"./videos/hand_test_02.mp4")
parser.add_argument(
    "--fps",
    default=30,
    type=int,
    help="frame-per-second used in saving video when works on webcam mode",
)
parser.add_argument("--save_path", default=r"./saves/camera.mp4")
opts = parser.parse_args()


def main():
    # For webcam input:
    if opts.camera:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, opts.fps)
    else:
        cap = cv2.VideoCapture(opts.inp)

    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    # capability 0 for the light model and 1 for large model
    tracker = HandsVisualTracker(capability=opts.capability, debug=opts.debug, threshold=0.9)
    writer = VideoWriter(opts.save_path, cap, (w, h))
    writer_ori = VideoWriter("./saves/source.mp4", cap, (w, h))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
            break

        if opts.camera:
            frame = cv2.flip(frame, flipCode=1)  # Flip the image horizontally for a later selfie-view display.
        canvas = tracker(frame)
        cv2.imshow("Window", canvas)
        writer.write(canvas)
        writer_ori.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            writer_ori.release()
            cap.release()
            break

    writer.release()
    writer_ori.release()
    cap.release()


if __name__ == "__main__":
    main()
