import cv2
import sys
import os
import time
import numpy as np
from argparse import ArgumentParser

pro_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pro_dir)

from lib.pose import PoseLandmarkTRT
from lib.utils.video import VideoWriter


parser = ArgumentParser()
parser.add_argument("--inp", default="./videos/whole_body_driver.mp4")
parser.add_argument(
    "--save_path",
    type=str,
    default=r"./saves",
    help="video file path to be saved",
)
opts = parser.parse_args()


def circle_st(image, x, y, radius, color):
    x, y = y, x
    color = np.array(color)
    width, height, c = image.shape
    region_expansion = 2
    x_left = round(x - radius) - region_expansion
    x_right = round(x + radius) + region_expansion
    y_top = round(y - radius) - region_expansion
    y_bottom = round(y + radius) + region_expansion

    x_left = int(min(max(x_left, 0), width - 1))
    x_right = int(min(max(x_right, 0), width - 1))
    y_top = int(min(max(y_top, 0), height - 1))
    y_bottom = int(min(max(y_bottom, 0), height - 1))

    points = []
    ratios = []
    for i in range(x_left, x_right + 1):
        for j in range(y_top, y_bottom + 1):
            dis = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
            if dis < radius + 0.5:
                points.append([i, j])
                ratios.append(min(radius + 0.5 - dis, 1.0))

    for idx, (xx, yy) in enumerate(points):
        image[xx, yy, :] = color * ratios[idx] + image[xx, yy, :] * (1 - ratios[idx])

    return image


def show_body_info(img, det_result, thickness=2, font_scale=0.7, font=cv2.FONT_HERSHEY_SIMPLEX):
    link_pairs_19 = [
        [15, 13],
        [13, 11],
        [11, 5],
        [12, 14],
        [14, 16],
        [12, 6],
        [3, 1],
        [1, 2],
        [1, 0],
        [0, 2],
        [2, 4],
        [9, 7],
        [7, 5],
        [5, 6],
        [6, 8],
        [8, 10],
        [9, 17],
        [10, 18],
    ]
    link_pairs_25 = [
        [15, 13],
        [13, 11],
        [11, 5],
        [12, 14],
        [14, 16],
        [12, 6],
        [3, 1],
        [1, 2],
        [1, 0],
        [0, 2],
        [2, 4],
        [9, 7],
        [7, 5],
        [5, 6],
        [6, 8],
        [8, 10],
        [9, 17],
        [10, 18],
        [15, 19],
        [15, 20],
        [15, 21],
        [16, 22],
        [16, 23],
        [16, 24],
    ]
    color = [
        (255.0, 0.0, 85.0),
        (255.0, 0.0, 0.0),
        (255.0, 85.0, 0.0),
        (255.0, 170.0, 0.0),
        (255.0, 255.0, 0.0),
        (170.0, 255.0, 0.0),
        (85.0, 255.0, 0.0),
        (0.0, 255.0, 0.0),
        (0.0, 255.0, 85.0),
        (0.0, 255.0, 170.0),
        (0.0, 255.0, 255.0),
        (0.0, 170.0, 255.0),
        (0.0, 85.0, 255.0),
        (0.0, 0.0, 255.0),
        (255.0, 0.0, 170.0),
        (170.0, 0.0, 255.0),
        (255.0, 0.0, 255.0),
        (85.0, 0.0, 255.0),
        (0.0, 125.0, 155.0),
        (0.0, 85.0, 255.0),
        (255.0, 170.0, 0.0),
        (255.0, 255.0, 0.0),
        (170.0, 255.0, 0.0),
        (85.0, 255.0, 0.0),
        (0.0, 255.0, 0.0),
        (0.0, 255.0, 85.0),
    ]

    if det_result.human_count > 0:
        for j in range(det_result.human_count):
            for land_id in range(det_result.d_humans[j].key_points_count):
                if det_result.d_humans[j].keypoints_score[land_id] > 0.5:
                    x = det_result.d_humans[j].points_array[land_id].x
                    y = det_result.d_humans[j].points_array[land_id].y
                    img = circle_st(img, x, y, 5, [255, 255, 0])
                    img = cv2.putText(
                        img,
                        str(land_id),
                        (int(x + 10), int(y + 10)),
                        font,
                        font_scale,
                        (0, 20, 229),
                        thickness,
                        cv2.LINE_AA,
                    )

            link_pairs = link_pairs_19
            if det_result.d_humans[j].key_points_count == 19:
                link_pairs = link_pairs_19
            elif det_result.d_humans[j].key_points_count == 25:
                link_pairs = link_pairs_25

            for idx, link in enumerate(link_pairs):
                if (
                    det_result.d_humans[j].keypoints_score[link[0]] > 0.5
                    and det_result.d_humans[j].keypoints_score[link[1]] > 0.5
                ):
                    joint_x1 = det_result.d_humans[j].points_array[link[0]].x
                    joint_y1 = det_result.d_humans[j].points_array[link[0]].y
                    joint_x2 = det_result.d_humans[j].points_array[link[1]].x
                    joint_y2 = det_result.d_humans[j].points_array[link[1]].y

                    start_point = (int(joint_x1), int(joint_y1))
                    end_point = (int(joint_x2), int(joint_y2))
                    cv2.line(img, start_point, end_point, color[idx], 2)
    return img


def main():
    cap = cv2.VideoCapture(opts.inp)

    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    writer = VideoWriter(os.path.join(opts.save_path, os.path.basename(opts.inp)), cap, (w, h))

    pose_model = PoseLandmarkTRT()
    counter = 0
    while True:
        print(f"counter: {counter}")

        frame_got, sample_frame = cap.read()
        if frame_got is False:
            break

        start_time = time.time()
        pred_result = pose_model(sample_frame)
        print("process time = %.6f ms" % (1000 * (time.time() - start_time)))
        show_result = show_body_info(sample_frame, pred_result)

        if writer is not None:
            writer.write(show_result)

        counter += 1

    pose_model.release()

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()
