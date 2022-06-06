import cv2
import os
import sys
import time
from argparse import ArgumentParser

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.utils.buffer import BufferPipe
from lib.utils.draw import BufferDraw
from lib.utils.video import VideoWriter
from lib.hands.hands_tracker_gesture import HandsTrackerGesture


parser = ArgumentParser()
parser.add_argument("--inp", default=r"./videos")
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug mode show bbox and some other information",
)
parser.add_argument("--buffer_size", type=int, default=30, help="buffer size for activating gesture effect")
parser.add_argument("--hits", type=int, default=5, help="number of hits for one buffer to show gesture effect")
parser.add_argument(
    "--fps",
    default=30,
    type=int,
    help="frame-per-second used in saving video when works on webcam mode",
)
parser.add_argument("--save_folder", default=r"./saves")
opts = parser.parse_args()


def main():
    video_paths = sorted(
        [
            os.path.join(opts.inp, file)
            for file in os.listdir(opts.inp)
            if file.endswith(".mp4") or file.endswith(".avi")
        ]
    )

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    video_folder = os.path.join(opts.save_folder, time_str)
    debug_folder = os.path.join(opts.save_folder, time_str, "debug")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    for video_path in video_paths:
        buffer = BufferPipe(opts.buffer_size, opts.hits)
        buffer_drawer = BufferDraw()

        print(f"video_path: {video_path}")

        cap = cv2.VideoCapture(video_path)
        tracker = HandsTrackerGesture(debug=opts.debug, is_remote=True)

        w, h = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = VideoWriter(
            os.path.join(video_folder, os.path.basename(video_path)),
            cap,
            (w, h),
        )

        counter = 0
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
                break

            canvas, is_det, predicts = tracker(frame)
            label = buffer.add(predicts, counter)
            canvas = buffer_drawer.draw(canvas, label)

            if is_det:
                cv2.imwrite(
                    os.path.join(
                        debug_folder,
                        os.path.basename(video_path) + f"_{str(counter).zfill(5)}.jpg",
                    ),
                    canvas,
                )

            writer.write(canvas)
            counter += 1

        writer.release()
        cap.release()


if __name__ == "__main__":
    main()
