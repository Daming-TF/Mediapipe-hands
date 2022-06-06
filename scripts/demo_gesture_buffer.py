import cv2
import os
import sys
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.hands.hands_tracker_gesture import HandsTrackerGesture
from lib.utils.draw import BufferDraw
from lib.utils.video import VideoWriter
from lib.utils.buffer import BufferPipe


parser = ArgumentParser()
parser.add_argument("--camera", action="store_true", help="open camera, default is test video")
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode show bbox and some other information",
)
parser.add_argument(
    "--pause",
    action="store_true",
    help="used to interpolating some frames to check gesture results",
)

parser.add_argument(
    "--roi_mode",
    type=int,
    default=1,
    choices=[0, 1],
    help="0: hand-detector, 1: pose-landmark for providing the position of hands",
)
parser.add_argument(
    "--capability",
    default=1,
    type=int,
    choices=[0, 1],
    help="model capability, 1 for large and 0 for lite hand-models",
)
parser.add_argument("--buffer_size", type=int, default=30, help="buffer size for activating gesture effect")
parser.add_argument("--hits", type=int, default=5, help="number of hits for one buffer to show gesture effect")
parser.add_argument("--inp", default=r"./videos/test_part_01.mp4")
parser.add_argument(
    "--fps",
    default=30,
    type=int,
    help="frame-per-second used in saving video when works on webcam mode",
)
parser.add_argument("--save_path", default=r"./saves/camera.mp4")
opts = parser.parse_args()


def main():
    buffer = BufferPipe(opts.buffer_size, opts.hits)
    buffer_drawer = BufferDraw()

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

    # capability 0 for the light model and 1 for large model
    tracker = HandsTrackerGesture(debug=opts.debug, capability=opts.capability, roi_mode=opts.roi_mode)
    writer = VideoWriter(
        opts.save_path if opts.camera else os.path.join(os.path.dirname(opts.save_path), os.path.basename(opts.inp)),
        cap,
        (w, h),
    )

    counter = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
            break

        if opts.camera:
            frame = cv2.flip(frame, flipCode=1)  # Flip the image horizontally for a later selfie-view display.

        canvas, is_det, predicts = tracker(frame)
        label = buffer.add(predicts, counter)
        canvas = buffer_drawer.draw(canvas, label)

        cv2.imshow("Window", canvas)
        writer.write(canvas)

        if opts.pause and is_det:
            if cv2.waitKey(0) & 0xFF == 27:
                writer.release()
                cap.release()
                break

        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            cap.release()
            break

        counter += 1

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
