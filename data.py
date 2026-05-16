import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import modcrop


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size):
        self.data = DIV2K(scale, path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 801)]  # TODO:use both train and valid (901)

        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_lr_x{}.npy".format(self.scale))
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f + "x{}.png".format(self.scale))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f + ".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        im = self.lr_ims[key]

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        # c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, :]
        im = im[i:i + self.sz, j:j + self.sz, :]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.transpose(lb.astype(np.float32) / 255.0, [2, 0, 1])
        im = np.transpose(im.astype(np.float32) / 255.0, [2, 0, 1])

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = 0  # Будем считать реальное количество изображений

        for dataset in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
            hr_folder = os.path.join(path, dataset, 'HR')
            if not os.path.exists(hr_folder):
                continue

            files = os.listdir(hr_folder)
            files.sort()
            self.files[dataset] = files
            _ims_all += len(files) * 2  # Учитываем HR и LR для каждого файла

            for i in range(len(files)):
                try:
                    # Загрузка HR изображения
                    hr_path = os.path.join(hr_folder, files[i])
                    im_hr = np.array(Image.open(hr_path))
                    im_hr = modcrop(im_hr, scale)
                    if len(im_hr.shape) == 2:
                        im_hr = np.expand_dims(im_hr, axis=2)
                        im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                    key = dataset + '_' + files[i][:-4]
                    self.ims[key] = im_hr

                    # Загрузка LR изображения
                    lr_path = os.path.join(path, dataset, 'LR_bicubic', f'X{scale}', files[i][:-4] + f'x{scale}.png')
                    im_lr = np.array(Image.open(lr_path))

                    if len(im_lr.shape) == 2:
                        im_lr = np.expand_dims(im_lr, axis=2)
                        im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                    # Проверка и корректировка размеров
                    if im_lr.shape[0] * scale != im_hr.shape[0]:
                        expected_size = im_lr.shape[0] * scale
                        im_hr = im_hr[:expected_size, :expected_size, :]

                    key = dataset + '_' + files[i][:-4] + f'x{scale}'
                    self.ims[key] = im_lr

                except Exception as e:
                    print(f"Error loading {files[i]}: {str(e)}")
                    _ims_all -= 2  # Уменьшаем счетчик при ошибке загрузки
                    continue

        # Убираем жесткую проверку, просто выводим информацию
        print(f"Loaded {len(self.ims.keys())} images (expected about {_ims_all})")