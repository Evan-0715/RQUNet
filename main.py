import os
import argparse
from src.model import RQUNet as network
from src.data_process import save_dataset
from src.util import show_pic, read_pics, save_pic

# python main.py --test_model --pic_show  --test_pic_num 010

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess", action="store_true", help="Need preprocessing?")
parser.add_argument("--dataset", default="BSD", help="Dataset need to be tested. (BSD or Kodak)")
parser.add_argument("--test_model", action="store_true", help="Test model?")
parser.add_argument("--sigma", type=int, default=20, help="Noise level. (10/20/30/40)")
parser.add_argument("--test_pic_num", default="003", help="Which No. of image to be tested.")
parser.add_argument("--pic_show", action="store_true", help="Display image?")
parser.add_argument("--pic_save", action="store_true", help="Save image?")
args = parser.parse_args()

PREPROCESS = args.preprocess
DATA_SET = {"BSD": "./data/BSD-100", "Kodak": "./data/Kodak-24"}[args.dataset]
DATA_BIN = {"BSD": "./bin/BSD-100", "Kodak": "./bin/Kodak-24"}[args.dataset]
TEST = args.test_model
SIGMA = args.sigma
PIC_NUM = args.test_pic_num
SHOW = args.pic_show
SAVE = args.pic_save
WEIGHT_PATH = {"BSD": "./model/BSD-100/", "Kodak": "./model/Kodak-24/"}[args.dataset] + 'model_weights_{}.h5'.format(SIGMA)
SAVE_PATH = {"BSD": "./fig/BSD-100", "Kodak": "./fig/Kodak-24"}[args.dataset]


if __name__ == '__main__':
    if PREPROCESS:
        if not os.path.exists(DATA_BIN):
            os.makedirs(DATA_BIN)
        print('Making binary file to {} ...'.format(DATA_BIN))
        save_dataset(DATA_SET, DATA_BIN + '/clean.npy', 0)
        save_dataset(DATA_SET, DATA_BIN + '/noise_10.npy', 10)
        save_dataset(DATA_SET, DATA_BIN + '/noise_20.npy', 20)
        save_dataset(DATA_SET, DATA_BIN + '/noise_30.npy', 30)
        save_dataset(DATA_SET, DATA_BIN + '/noise_40.npy', 40)
        print('Done!')
    if TEST:
        clean_pic, noise_pic = read_pics(DATA_SET, PIC_NUM, SIGMA)
        model = network()
        model.load_weights(WEIGHT_PATH)
        if SHOW:
            show_pic(model, clean_pic, noise_pic)

        if SAVE:
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            save_pic(model, clean_pic, noise_pic, SAVE_PATH + "/noise_{}_{}.png".format(SIGMA, PIC_NUM))
