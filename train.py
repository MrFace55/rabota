import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

from models import *
from data import Provider, SRBenchmark, degrade_image
from utils import PSNR, _rgb2ycbcr, seed_everything

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sanitize_name(name):
    """Заменяет запрещенные символы Windows на подчеркивания"""
    return name.replace(':', '_').replace('/', '_').replace('\\', '_')


def parse_args():
    parser = argparse.ArgumentParser("Training Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--train-dir", type=str, default='./data/train/DIV2K',
                        help="Training images")
    parser.add_argument("--val-dir", type=str, default='./data/test/',
                        help="Validation images")
    parser.add_argument("--i-display", type=int, default=500,
                        help="display info every N iteration")
    parser.add_argument("--i-validate", type=int, default=500,
                        help="validation every N iteration")
    parser.add_argument("--i-save", type=int, default=2000,
                        help="save checkpoints every N iteration")

    parser.add_argument("--upscale", nargs='+', type=int, default=[1],
                        help="upscaling factors (1 for color correction)")
    parser.add_argument("--crop-size", type=int, default=48,
                        help="input LR training patch size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Set 0 for from scratch, else will load saved params and trains further")
    parser.add_argument("--train-iter", type=int, default=200000,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0, help='weight decay')

    parser.add_argument('--degradation', type=str, default='gaussian',
                        choices=['gaussian', 'blur', 'gaussian_blur', 'jpeg', 'mixed', 'mixed_with_blur', 'none'],
                        help='degradation type for color correction')
    parser.add_argument('--noise-std', type=float, default=25,
                        help='Gaussian noise standard deviation')
    parser.add_argument('--jpeg-quality', type=int, default=75,
                        help='JPEG compression quality')
    parser.add_argument('--kernel-size', type=int, default=5,
                        help='Gaussian blur kernel size')
    parser.add_argument('--blur-sigma', type=float, default=0,
                        help='Gaussian blur sigma (0 = auto)')

    parser.add_argument('--msb', type=str, default='hdb', choices=['hdb', 'hd', 'hdt'])
    parser.add_argument('--lsb', type=str, default='hd', choices=['hdb', 'hd', 'hdt'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    # Создаем имя эксперимента
    raw_exp_name = "msb:{}-lsb:{}-act:{}-nf:{}-{}-deg:{}".format(
        args.msb, args.lsb, args.act_fn, args.n_filters, factors, args.degradation)

    # Сохраняем оригинальное имя для логов внутри файла, но используем безопасное для путей
    args.exp_name = raw_exp_name
    args.safe_exp_name = sanitize_name(raw_exp_name)

    act_fn_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]

    # Always use new implementation with HDTBLUT support
    from models import HKNet
    print("Using NEW implementation (hklut_1.py, luts_1.py, hknet_1.py, units_1.py with HDTBLUT/HDTUnit support)")

    return args


def SaveCheckpoint(models, opt_G, i, args, best=False):
    # Используем безопасное имя для путей
    exp_dir = 'checkpoint/{}'.format(args.safe_exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    if best:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), '{}/model_G_S{}_best.pth'.format(exp_dir, stage))
            else:
                torch.save(model.state_dict(), '{}/model_G_S{}_best.pth'.format(exp_dir, stage))
        torch.save(opt_G.state_dict(), '{}/opt_G_best.pth'.format(exp_dir))
        print("Best checkpoint saved to {}".format(exp_dir))
    else:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), '{}/model_G_S{}_i{:06d}.pth'.format(exp_dir, stage, i))
            else:
                torch.save(model.state_dict(), '{}/model_G_S{}_i{:06d}.pth'.format(exp_dir, stage, i))
        torch.save(opt_G.state_dict(), '{}/opt_G_i{:06d}.pth'.format(exp_dir, i))
        print("Checkpoint saved {}".format(str(i)))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed_everything(args.seed)

    ### Tensorboard for monitoring ###
    # Используем безопасное имя для директории логов
    writer = SummaryWriter(log_dir='./log/{}'.format(args.safe_exp_name))

    models = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    for s in args.upscale:
        models.append(HKNet(msb=args.msb, lsb=args.lsb, nf=args.n_filters, upscale=s, act=args.act_fn).to(device))

    ## Optimizers
    opt_G = optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters()))} for model in models],
                       lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, eps=1e-8, amsgrad=False)

    scheduler = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[100000, 150000], gamma=0.1)

    ## Load saved params
    if args.start_iter > 0:
        exp_dir = 'checkpoint/{}'.format(args.safe_exp_name)
        for stage in range(n_stages):
            lm = torch.load('{}/model_G_S{}_i{:06d}.pth'.format(exp_dir, stage, args.start_iter))
            models[0].load_state_dict(lm, strict=True)

        lm = torch.load('{}/opt_G_i{:06d}.pth'.format(exp_dir, args.start_iter))
        opt_G.load_state_dict(lm)

    if torch.cuda.device_count() > 1:
        models = [nn.DataParallel(model) for model in models]

    # Prepare degradation parameters
    degradation_params = {
        'sigma': args.noise_std, 
        'quality': args.jpeg_quality,
        'kernel_size': args.kernel_size,
        'blur_sigma': args.blur_sigma
    }

    # Training dataset
    train_loader = Provider(args.batch_size, args.n_workers, sr_scale, args.train_dir,
                            args.crop_size, degradation=args.degradation,
                            degradation_params=degradation_params)

    # Validation dataset
    # Для валидации загружаем чистые изображения (GT).
    # Деградацию применим вручную в цикле валидации.
    valid_loader = SRBenchmark(args.val_dir, scale=1)  # scale=1, так как размеры совпадают
    valid_datasets = ['Set5', 'Set14']  # Добавили Set14 для проверки

    ## Prepare directories
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/{}'.format(args.safe_exp_name)):
        os.mkdir('checkpoint/{}'.format(args.safe_exp_name))
    if not os.path.isdir('log'):
        os.mkdir('log')

    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    ### TRAINING
    best_psnr = 0.0
    for i in tqdm(range(args.start_iter + 1, args.train_iter + 1)):

        for model in models:
            model.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = train_loader.next()
        batch_H = batch_H.to(device)  # BxCxHxW, range [0,1]
        batch_L = batch_L.to(device)  # BxCxHxW, range [0,1] (уже деградировано в Provider)

        dT += time.time() - st

        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        x = batch_L
        for model in models:
            x = model(x)

        # Модель уже возвращает результат с residual connection и clamp
        pred = x
        loss_G = F.mse_loss(pred, batch_H)

        # Update
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += args.batch_size
        l_accum[0] += loss_G.item()

        ## Show information
        if i % args.i_display == 0:
            writer.add_scalar('loss_Pixel', l_accum[0] / args.i_display, i)
            print("{}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                args.exp_name, i, accum_samples, l_accum[0] / args.i_display, dT / args.i_display, rT / args.i_display))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        ## Save models
        if i % args.i_save == 0:
            SaveCheckpoint(models, opt_G, i, args)

        ## Validation
        if i % args.i_validate == 0:
            with torch.no_grad():
                for model in models:
                    model.eval()

                for j in range(len(valid_datasets)):
                    psnrs = []
                    files = valid_loader.files.get(valid_datasets[j], [])

                    if not files:
                        print(f"Dataset {valid_datasets[j]} not found in validation loader.")
                        continue

                    for k in range(len(files)):
                        fname = files[k][:-4]  # remove .png/.jpg
                        key = valid_datasets[j] + '_' + fname

                        # Загружаем GT (чистое изображение)
                        if key not in valid_loader.ims:
                            continue

                        img_gt = valid_loader.ims[key]  # (H, W, 3) range [0, 255]

                        # === КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ДЛЯ ЦВЕТОКОРРЕКЦИИ ===
                        # Генерируем деградированный вход из GT прямо здесь
                        img_degraded = degrade_image(img_gt, args.degradation, **degradation_params)

                        # Нормализация и подготовка тензора
                        input_im = img_degraded.astype(np.float32) / 255.0
                        val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(
                            device)  # (1, 3, H, W)

                        x = val_L
                        for model in models:
                            x = model(x)

                        # Модель уже возвращает результат с residual connection и clamp
                        x = x

                        # Output
                        image_out = x.cpu().data.numpy()
                        image_out = np.transpose(image_out[0], [1, 2, 0])  # HxWxC
                        image_out = np.clip(image_out * 255.0, 0, 255).astype(np.uint8)

                        # PSNR on Y channel
                        h, w, _ = img_gt.shape
                        out_h, out_w, _ = image_out.shape

                        min_h, min_w = min(h, out_h), min(w, out_w)

                        psnrs.append(PSNR(_rgb2ycbcr(img_gt[:min_h, :min_w])[:, :, 0],
                                          _rgb2ycbcr(image_out[:min_h, :min_w])[:, :, 0], 1))

                    if len(psnrs) > 0:
                        mean_psnr = np.mean(np.asarray(psnrs))

                        # save best psnr
                        if mean_psnr > best_psnr:
                            best_psnr = mean_psnr
                            SaveCheckpoint(models, opt_G, i, args, best=True)

                        print('Iter {} | Dataset {} | AVG Val PSNR: {:.2f}'.format(i, valid_datasets[j], mean_psnr))
                        writer.add_scalar('PSNR_valid/{}'.format(valid_datasets[j]), mean_psnr, i)
                    else:
                        print(f'Iter {i} | Dataset {valid_datasets[j]} | No images processed')

                    writer.flush()

    print(f'Best PSNR: {best_psnr}')