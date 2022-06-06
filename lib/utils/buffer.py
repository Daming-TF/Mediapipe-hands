import numpy as np


class BufferPipe(object):
    def __init__(self, size, hits, target_labels, num_classes):
        self.size = size
        self.hits = hits
        self.buffer = -1 * np.ones((2, self.size), dtype=np.uint8)

        self.target_lables = target_labels
        self.num_classes = num_classes

    def reset(self):
        self.buffer = -1 * np.ones((2, self.size), dtype=np.uint8)

    def add(self, preds, counter):
        for i, pred in enumerate(preds):
            self.buffer[i, counter % self.size] = pred

        max_counts = [0, 0]
        activate_label = [-1, -1]

        for i in range(len(activate_label)):  # left-buffer and right-buffer
            for target_label in self.target_lables:
                num_hits = np.sum(self.buffer[i, :] == target_label)

                if (num_hits > max_counts[i]) and (num_hits > self.hits):
                    max_counts[i] = num_hits
                    activate_label[i] = target_label

        return activate_label
