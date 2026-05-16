# import numpy as np
# sample = np.load("D:/PycharmProjects/NIR/luts/msb-hdb-lsb-hd-act-gelu-nf-64-2/S0_MSB_HDB_H_x2_4bit_int8.npy")
# print(sample.shape, sample.dtype)
from PIL import Image
import os

hr_dir = "D:/PycharmProjects/NIR/data/test/Set5/HR"
lr_dir = "D:/PycharmProjects/NIR/data/test/Set5/LR_bicubic/X4"

for hr_file in os.listdir(hr_dir):
    hr_path = os.path.join(hr_dir, hr_file)
    lr_path = os.path.join(lr_dir, hr_file[:-4] + 'x4.png')

    with Image.open(hr_path) as hr_img:
        lr_size = (hr_img.width // 4, hr_img.height // 4)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        lr_img.save(lr_path)
        print(f"Resized {hr_file} from {hr_img.size} to {lr_size}")