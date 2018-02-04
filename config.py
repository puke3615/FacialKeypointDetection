import os

# 根路径配置
PATH_ROOT = 'G:/Dataset/FacialKeyPointDetection'

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
