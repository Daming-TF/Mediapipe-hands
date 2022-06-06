import cv2
import os


class VideoWriter(object):
    def __init__(self, save_name, cap, size=None):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        # parameters of the video header
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.img_size = size if size is not None else frame_size
        codec = self.get_codec(save_name)

        self.writer = cv2.VideoWriter(save_name, codec, fps, self.img_size)

    @staticmethod
    def get_codec(save_name):
        _, ext = os.path.splitext(save_name)
        if ext == ".avi":
            codec = cv2.VideoWriter_fourcc(*"XVID")
        elif ext == ".mp4":
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            # codec = cv2.VideoWriter_fourcc(*"avc1")
        else:
            raise Exception(" [!] Video extension of {} is not supported!".format(ext))

        return codec

    def write(self, frame):
        if (frame.shape[0] != self.img_size[1]) or (frame.shape[1] != self.img_size[0]):  # self.img_size: (w, h)
            frame = cv2.resize(
                frame,
                (self.img_size[0], self.img_size[1]),
                interpolation=cv2.INTER_CUBIC,
            )
        self.writer.write(frame)

    def release(self):
        self.writer.release()
