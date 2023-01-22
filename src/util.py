'''
Tool functions
'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow import image
from keras.losses import mean_squared_error
from src.data_process import read_image, add_noise

subfig_scale = 64
scale = 512
subfig_num = (scale // subfig_scale) ** 2


def rebuild_pic_3_channel(one_pic):
    return np.concatenate(
        [
            np.concatenate(one_pic[i:i + scale // subfig_scale], axis=1)
            for i in range(0, subfig_num, scale // subfig_scale)
        ],
        axis=0,
    )


def psnr_pred(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)


def ssim_pred(y_true, y_pred):
    return image.ssim(y_true, y_pred, max_val=1.0)


def read_pics(DATA_SET, PIC_NUM, SIGMA):
    clean_pic = read_image('{}/{}.png'.format(DATA_SET, PIC_NUM))
    clean_pic1 = read_image('{}/{}.png'.format(DATA_SET, PIC_NUM))
    noise_pic = add_noise(clean_pic1, SIGMA)
    clean_pic = clean_pic / 255
    noise_pic = noise_pic / 255
    return clean_pic, noise_pic


def show_pic(model, clean_pic, noise_pic):
    model.compile(optimizer="Adam", loss=mean_squared_error, metrics=[psnr_pred, ssim_pred])
    noise_pic_1 = noise_pic[np.newaxis, :, :, :]
    predict_unsqueeze = model.predict(noise_pic_1, verbose=1)
    predict1 = predict_unsqueeze.squeeze()
    predict = np.clip(predict1, 0, 1)
    plt.subplot(1, 3, 1)
    plt.imshow(noise_pic)
    plt.title('Noisy Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(clean_pic)
    plt.title('Gt Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predict)
    plt.title('Denoised Image')
    plt.axis('off')
    plt.show()

def save_pic(model, clean_pic, noise_pic, path):
    model.compile(optimizer="Adam", loss=mean_squared_error, metrics=[psnr_pred, ssim_pred])
    noise_pic_1 = noise_pic[np.newaxis, :, :, :]
    predict_unsqueeze = model.predict(noise_pic_1, verbose=1)
    predict1 = predict_unsqueeze.squeeze()
    predict = np.clip(predict1, 0, 1)
    plt.subplot(1, 3, 1)
    plt.imshow(noise_pic)
    plt.title('Noisy Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(clean_pic)
    plt.title('Gt Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predict)
    plt.title('Denoised Image')
    plt.axis('off')
    plt.savefig(path)
    plt.close()
