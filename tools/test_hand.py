import cv2
import os
import sys

pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)

from lib.hands.detector import HandDetModel


def main():
    cap = cv2.VideoCapture(0)
    model = HandDetModel()

    while True:
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        img_show = frame.copy()
        hand_boxes = model(frame)
        # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
        for hand_box in hand_boxes:
            cv2.rectangle(
                img_show,
                pt1=(int(hand_box[0][0]), int(hand_box[0][1])),
                pt2=(int(hand_box[1][0]), int(hand_box[1][1])),
                color=(0, 0, 255),
                thickness=2,
            )
        print(f"len(hand_boxes): {len(hand_boxes)}")

        cv2.imshow("Window", img_show)
        if cv2.waitKey(1) == 27:
            exec("Esc clicked!")


if __name__ == "__main__":
    main()
