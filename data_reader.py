from PIL import Image
from config import *
from utils import *
import numpy as np
import cv2
import os


class DataReader:
    def __init__(self, path, batch_size=32, shuffle=True, im_size=50, x_mode=None, y_mode=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.im_size = im_size
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.parse(path)
        self.epoch = 1
        self.index_pool = np.full([len(self.files), 2], self.epoch)
        self.index_pool[:, 0] = range(len(self.files))

    def _read_batch(self, index_array):
        batch_files = self.files[index_array]
        if self.x_mode == 'file':
            batch_x = batch_files
        else:
            batch_x = np.zeros([len(index_array), self.im_size, self.im_size, 3])
            for i, file in enumerate(batch_files):
                im = Image.open(file)
                im = im.resize([self.im_size, self.im_size])
                batch_x[i, :] = im
        if self.y_mode == 'coord':
            batch_y = self.data_list[index_array][:, :4].astype(np.int)
        else:
            batch_y = self.data_list[index_array]
        return batch_x, batch_y

    def get_generator(self):
        index_getter = lambda: self.index_pool[self.index_pool[:, 1] == self.epoch][:, 0]
        index_list = index_getter()
        pool_size = len(index_list)
        if pool_size < self.batch_size:
            self.index_pool[self.index_pool[:, 1] != self.epoch] = self.epoch
            self.epoch += 1
            index_list = index_getter()
        if self.shuffle:
            index_array = np.random.choice(index_list, self.batch_size, replace=False)
        else:
            index_array = self.index_pool[:self.batch_size, 0]
        self.index_pool[index_array, 1] = self.epoch + 1
        yield self._read_batch(index_array)

    def parse(self, path):
        if not os.path.exists(path):
            raise Exception()
        with open(path) as f:
            files = []
            data_list = []
            for line in f:
                items = line.split(' ')
                files.append(os.path.join(PATH_ROOT, items[0].replace('\\', '/')))
                data_list.append([float(item) for item in items[1:]])
            self.files = np.array(files)
            self.data_list = np.array(data_list)


if __name__ == '__main__':
    reader = DataReader(PATH_TRAIN, batch_size=100, x_mode='file')
    batch_x, batch_y = next(reader.get_generator())
    print('Read %s images' % len(batch_x))

    index = 0
    show_mode = 0
    im_getter = lambda: (cv2.imread(batch_x[index]), batch_y[index])
    show = lambda img: cv2.imshow('Image', img)
    while True:
        if index < 0:
            index = len(batch_x) - 1
        elif index == len(batch_x):
            index = 0
        im, info = im_getter()
        print(index, batch_x[index])
        show(show_keypoint(im, info, show_mode))
        key = cv2.waitKey() & 0xFF
        if key == ord('a'):
            index -= 1
        elif key == ord('d'):
            index += 1
        elif key == ord('s'):
            show_mode = 0
        elif key == ord('w'):
            show_mode = 1
        elif key == ord('e'):
            show_mode = 2
        elif key == ord('r'):
            show_mode = 3
        elif key == ord('q'):
            break
