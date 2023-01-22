import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Compress image to (SCALE, SCALE, 3), than split into (SUBFIG_SCALE, SUBFIG_SCALE, 3)
than reshape every subfig into a one-row vertor
"""

SUBFIG_SCALE = 64
SCALE = 512
n = (SCALE // SUBFIG_SCALE) ** 2


def split_image(img):
    '''
    img: np.ndarray
    '''
    row, col, _ = img.shape
    res = []
    for i in range(0, row, SUBFIG_SCALE):
        for j in range(0, col, SUBFIG_SCALE):
            res.append(img[i: i + SUBFIG_SCALE, j: j + SUBFIG_SCALE, :].flatten())
    return np.array(res)


def add_noise(img, sigma):
    '''
    Add Guassian noise into image
    '''
    if sigma != 0:
        img += np.random.normal(0, sigma, img.shape).astype(int)
    # clip函数限制数组大小范围
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def read_image(pic_path):
    '''
    input: directory path of image
    output: np.ndarray
    '''
    img = cv.imread(pic_path)
    img = cv.resize(img, (SCALE, SCALE), cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR).astype(int)
    return img


def rebuild_picture(src):
    """
    Re-construct image
    """

    def inner(one_pic):
        temp_list = [i.reshape(SUBFIG_SCALE, SUBFIG_SCALE, 3) for i in one_pic]
        return np.concatenate([
            np.concatenate(
                temp_list[i: i + SCALE // SUBFIG_SCALE],
                axis=1
            ) for i in range(0, n, SCALE // SUBFIG_SCALE)
        ], axis=0, )

    pics = np.load(src)
    for i in range(0, len(pics), n):
        img = inner(pics[i: i + n, :])
        plt.imshow(img)
        plt.show()


def save_dataset(input_path, output_path, sigma=0):
    '''
    Saving image data to .npy files
'''
    np.save(output_path, np.concatenate([split_image(
        add_noise(
            read_image('{}/{}'.format(input_path, pic)),
            sigma
        )
    ) for pic in os.listdir(input_path)]))


def load_data(path, sigma):
    test_clean = np.load("{}/clean.npy".format(path)).reshape(
        (-1, SUBFIG_SCALE, SUBFIG_SCALE, 3)
    ).astype(np.float32) / 255
    test_noise = np.load("{}/noise_{}.npy".format(path, sigma)).reshape(
        (-1, SUBFIG_SCALE, SUBFIG_SCALE, 3)
    ).astype(np.float32) / 255

    return test_clean, test_noise


if __name__ == "__main__":
    pass
