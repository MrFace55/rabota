import os
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from data import SRBenchmark
from utils import PSNR, cal_ssim, _rgb2ycbcr
from PIL import Image
from utils import seed_everything
from models import HKLUT

device = 'cpu'


def parse_args():
    parser = argparse.ArgumentParser("Testing Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--test-dir", type=str, default='./data/test/',
                        help="Testing images")
    parser.add_argument("--lut-dir", type=str, default='./luts',
                        help="Directory for storing cached LUTs")
    parser.add_argument("--result-dir", type=str, default='./results',
                        help="Directory to store resulted images")
    parser.add_argument("--upscale", nargs='+', type=int, default=[2, 2],
                        help="upscaling factors")
    parser.add_argument('--msb', type=str, default='hdb', choices=['hdb', 'hd'])
    parser.add_argument('--lsb', type=str, default='hd', choices=['hdb', 'hd'])
    parser.add_argument('--act-fn', type=str, default='gelu',
                        choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64,
                        help="number of filters in intermediate layers")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb-{}-lsb-{}-act-{}-nf-{}-{}".format(
        args.msb, args.lsb, args.act_fn, args.n_filters, factors)
    args.lut_path = str(Path(args.lut_dir) / args.exp_name)
    return args


def create_safe_dir(path):
    """Создает директорию, заменяя недопустимые символы"""
    safe_path = str(Path(path))
    os.makedirs(safe_path, exist_ok=True)
    return safe_path


def load_lut_file(lut_dir, stage, prefix, lut_type, scale):
    """Безопасная загрузка LUT-файла с правильным преобразованием типов"""
    pattern = f"S{stage}_{prefix}_{lut_type}_x{scale}_4bit_int8.npy"
    file_path = Path(lut_dir) / pattern

    if not file_path.exists():
        available = [f.name for f in Path(lut_dir).glob('*.npy')]
        raise FileNotFoundError(
            f"LUT file {pattern} not found. Available files: {available}"
        )

    # Правильная последовательность преобразований:
    numpy_array = np.load(file_path)
    return torch.from_numpy(numpy_array.astype(np.int32))  # Используем np.int32 вместо np.int_


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # Нормализация путей для Windows
    args.lut_path = str(Path(args.lut_path).resolve())
    args.test_dir = str(Path(args.test_dir).resolve())
    args.result_dir = str(Path(args.result_dir).resolve())

    print("Resolved LUT path:", args.lut_path)
    print("Resolved test dir:", args.test_dir)

    # Создание директорий результатов
    result_exp_dir = create_safe_dir(Path(args.result_dir) / args.exp_name)
    print("Results will be saved to:", result_exp_dir)

    # Проверка существования директории LUT
    if not Path(args.lut_path).exists():
        raise FileNotFoundError(f"LUT directory not found: {args.lut_path}")

    # Проверка тестовых данных
    if not Path(args.test_dir).exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    luts = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)
    models = []

    # Загрузка LUT с обработкой ошибок
    try:
        for stage in range(n_stages):
            # MSB weights
            msb_weights = [
                load_lut_file(args.lut_path, stage, f"MSB_{args.msb.upper()}", ktype.upper(), args.upscale[stage])
                for ktype in args.msb
            ]

            # LSB weights
            lsb_weights = [
                load_lut_file(args.lut_path, stage, f"LSB_{args.lsb.upper()}", ktype.upper(), args.upscale[stage])
                for ktype in args.lsb
            ]

            models.append(HKLUT(
                msb_weights, lsb_weights,
                msb=args.msb, lsb=args.lsb,
                upscale=args.upscale[stage]
            ).to(device))
    except Exception as e:
        print(f"Error loading LUTs: {str(e)}")
        raise

    # Тестовые данные
    test_loader = SRBenchmark(args.test_dir, scale=sr_scale)
    test_datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

    # Тестирование
    with torch.no_grad():
        for model in models:
            model.eval()

        for dataset in test_datasets:
            psnrs = []
            ssims = []

            if dataset not in test_loader.files:
                print(f"Warning: Dataset {dataset} not found, skipping")
                continue

            for file in test_loader.files[dataset]:
                try:
                    key = f"{dataset}_{file[:-4]}"
                    img_gt = test_loader.ims[key]
                    input_im = test_loader.ims[f"{key}x{sr_scale}"]

                    # Преобразование изображения
                    input_im = input_im.astype(np.float32) / 255.0
                    val_L = torch.tensor(
                        np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)
                    ).to(device)

                    # Обработка через модели
                    x = val_L
                    for model in models:
                        x = model(x)

                    # Сохранение результата
                    image_out = np.clip(x.cpu().numpy()[0] * 255, 0, 255).astype(np.uint8)
                    image_out = np.transpose(image_out, [1, 2, 0])
                    output_path = Path(result_exp_dir) / f"{key}.png"
                    Image.fromarray(image_out).save(str(output_path))

                    # Расчет метрик с выравниванием размеров
                    y_gt = _rgb2ycbcr(img_gt)[:, :, 0]
                    y_out = _rgb2ycbcr(image_out)[:, :, 0]

                    min_height = min(y_gt.shape[0], y_out.shape[0])
                    min_width = min(y_gt.shape[1], y_out.shape[1])
                    y_gt = y_gt[:min_height, :min_width]
                    y_out = y_out[:min_height, :min_width]

                    psnrs.append(PSNR(y_gt, y_out, sr_scale))
                    ssims.append(cal_ssim(y_gt, y_out))

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
            if psnrs:
                print(f'Dataset {dataset} | AVG PSNR: {np.mean(psnrs):.2f} SSIM: {np.mean(ssims):.4f}')