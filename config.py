# coding=utf-8
import os
import platform

os_name = platform.system().lower()


def is_mac():
    return os_name.startswith('darwin')


def is_windows():
    return os_name.startswith('windows')


def is_linux():
    return os_name.startswith('linux')


# 根路径配置
if is_mac():
    PATH_ROOT = '/Users/zijiao/Documents/Dataset/FacialKeyPointDetection'
elif is_windows():
    PATH_ROOT = 'G:/Dataset/FacialKeyPointDetection'
else:
    raise EnvironmentError('No support for this os.')

# 训练集
PATH_TRAIN = os.path.join(PATH_ROOT, 'trainImageList.txt')
# 测试集
PATH_TEST = os.path.join(PATH_ROOT, 'testImageList.txt')


def load_data(path_data_set):
    with open(path_data_set) as f:
        data_set = [line for line in f if line.strip() != '']
        return data_set


if __name__ == '__main__':
    train_data = load_data(PATH_TRAIN)
    test_data = load_data(PATH_TEST)
    print('Train data length is %d.' % (len(train_data)))
    print('Test data length is %d.' % (len(test_data)))
