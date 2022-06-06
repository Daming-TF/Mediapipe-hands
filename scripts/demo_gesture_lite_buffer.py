import cv2
import os
import sys
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.hands.hands_tracker_gesture import HandsTrackerGestureLite
from lib.utils.video import VideoWriter


parser = ArgumentParser()
parser.add_argument("--camera", action="store_true", help="open camera, default is test video")
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode show bbox and some other information",
)

parser.add_argument(
    "--buffer_size",
    type=int,
    default=20,
    help="buffer size for activating gesture effect",
)
parser.add_argument(
    "--hits",
    type=int,
    default=5,
    help="number of hits for one buffer to show gesture effect",
)
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
    tracker = HandsTrackerGestureLite(buffer_size=opts.buffer_size, hits=opts.hits, debug=opts.debug)
    writer = VideoWriter(
        opts.save_path if opts.camera else os.path.join(os.path.dirname(opts.save_path), os.path.basename(opts.inp)),
        cap,
        (w, h),
    )

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")
            break

        if opts.camera:
            frame = cv2.flip(frame, flipCode=1)  # Flip the image horizontally for a later selfie-view display.

        pred_labels, label_names = tracker.run(frame, is_print=True)
        for i, pred_label in enumerate(pred_labels):
            if pred_label != -1:
                print(f"pred_label: {pred_label}, label_name: {label_names[i]}")

        canvas = tracker.drawer.get_canvas()
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
