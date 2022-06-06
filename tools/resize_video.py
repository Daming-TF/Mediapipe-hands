import cv2
import os
import sys

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.utils.video import VideoWriter


def main():
    video_path = r"/videos/test_part.mp4"
    save_path = r"/videos/resized_test_part.mp4"

    cap = cv2.VideoCapture(video_path)
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    writer = VideoWriter(save_path, cap, (int(0.5 * w), int(0.5 * h)))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")  # If loading a video, use 'break' instead of 'continue'.
            break

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Window", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            cap.release()
            break

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
