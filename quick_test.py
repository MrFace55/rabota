import torch
import numpy as np
import os
from PIL import Image
from models import HKNet
from utils import PSNR, _rgb2ycbcr
from data import degrade_image
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Параметры
WEIGHTS_PATH = './checkpoint/msb_hdb-lsb_hdb-act_relu-nf_48-1-deg_gaussian/model_G_S0_i003000.pth'
DATA_DIR = './data/Set14/HR'  # Путь к чистым изображениям
NOISE_STD = 25
UPSACLE = 1
MSB = 'hdb'
LSB = 'hdb'
NF = 48

print(f"Loading model from {WEIGHTS_PATH}...")
model = HKNet(msb=MSB, lsb=LSB, nf=NF, upscale=UPSACLE, act=torch.nn.ReLU)
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
# Обработка возможных префиксов
try:
    model.load_state_dict(state_dict)
except:
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)

model.to(device).eval()
print("Model loaded.")

# Поиск изображений
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
files.sort()

psnr_list = []
ssim_list = []  # Нужно будет добавить функцию SSIM если её нет в utils, пока пропустим или используем простую
de_list = []

print(f"Processing {len(files)} images from {DATA_DIR}...")

for fname in files:
    # Загрузка GT
    img_gt = np.array(Image.open(os.path.join(DATA_DIR, fname)).convert('RGB'))

    # Деградация (шум)
    img_noisy = degrade_image(img_gt, 'gaussian', sigma=NOISE_STD)

    # Подготовка тензора
    input_tensor = torch.from_numpy(np.transpose(img_noisy.astype(np.float32) / 255.0, [2, 0, 1])).unsqueeze(0).to(
        device)

    # Инференс
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        infer_time = time.time() - start

    # Пост-процессинг
    out_np = np.clip(output.cpu().squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255
    out_uint8 = out_np.astype(np.uint8)

    # Метрики
    # PSNR (Y channel)
    psnr_val = PSNR(_rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(out_uint8)[:, :, 0], 1)
    psnr_list.append(psnr_val)

    # Простой MSE для примера, если SSIM нет
    mse = np.mean((img_gt.astype(float) - out_uint8.astype(float)) ** 2)
    ssim_approx = 1.0 / (1.0 + mse / 10000)  # Заглушка, лучше использовать real SSIM

    print(f"{fname}: PSNR={psnr_val:.2f} dB, Time={infer_time * 1000:.2f}ms")

print("-" * 30)
print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Total Images: {len(files)}")