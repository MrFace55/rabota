import torch
import numpy as np
import os
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000

# Импорт ваших локальных модулей
from models import HKNet
from data import degrade_image
from utils import _rgb2ycbcr  # Для PSNR по яркости

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= НАСТРОЙКИ =================
WEIGHTS_PATH = './checkpoint/msb_hdb-lsb_hdb-act_relu-nf_48-1-deg_gaussian/model_G_S0_i003000.pth'
DATA_DIR = './data/Set14/HR'
SAVE_DIR = './results_comparison'
NOISE_STD = 25
MSB = 'hdb'
LSB = 'hdb'
NF = 48
UPSACLE = 1


# =============================================

def calculate_psnr(img1, img2):
    """PSNR по каналу Y (яркость)"""
    y1 = _rgb2ycbcr(img1)[:, :, 0]
    y2 = _rgb2ycbcr(img2)[:, :, 0]
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """SSIM по каналу Y"""
    y1 = _rgb2ycbcr(img1)[:, :, 0]
    y2 = _rgb2ycbcr(img2)[:, :, 0]
    return ssim(y1, y2, data_range=255)


def calculate_delta_e(img1, img2):
    """ΔE CIEDE2000 (среднее по всем пикселям)"""
    # Нормализация для skimage (0..1)
    lab1 = rgb2lab(img1.astype(np.float32) / 255.0)
    lab2 = rgb2lab(img2.astype(np.float32) / 255.0)
    # deltaE_cie2000 возвращает массив разниц, берем среднее
    de_map = deltaE_ciede2000(lab1, lab2)
    return np.mean(de_map)


def create_comparison_grid(gt, noisy, restored, filename):
    """Создает картинку [GT | Noisy | Restored]"""
    h, w, _ = gt.shape

    # Создаем пустое изображение шириной 3*w
    grid = np.zeros((h, w * 3, 3), dtype=np.uint8)

    grid[:, :w, :] = gt
    grid[:, w:w * 2, :] = noisy
    grid[:, w * 2:, :] = restored

    # Добавим подписи (опционально, здесь просто сохраняем сетку)
    pil_grid = Image.fromarray(grid)
    pil_grid.save(os.path.join(SAVE_DIR, f"comp_{filename}"))


print(f"Loading model from {WEIGHTS_PATH}...")
model = HKNet(msb=MSB, lsb=LSB, nf=NF, upscale=UPSACLE, act=torch.nn.ReLU)

# Загрузка весов с обработкой возможных ошибок ключей
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    # Убираем префикс 'module.' если он есть
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)

model.to(device).eval()
print("Model loaded successfully.")

# Подготовка папки результатов
os.makedirs(SAVE_DIR, exist_ok=True)

# Поиск изображений
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png') or f.endswith('.jpg')]
files.sort()

if not files:
    print(f"ERROR: No images found in {DATA_DIR}")
    exit()

print(f"Processing {len(files)} images from {DATA_DIR}...\n")

metrics = {'psnr': [], 'ssim': [], 'delta_e': [], 'time': []}

for fname in files:
    base_name = os.path.splitext(fname)[0]

    # 1. Загрузка GT
    img_gt = np.array(Image.open(os.path.join(DATA_DIR, fname)).convert('RGB'))

    # 2. Генерация шума (Noisy)
    # Фиксируем seed для воспроизводимости картинки шума
    np.random.seed(42)
    img_noisy = degrade_image(img_gt, 'gaussian', sigma=NOISE_STD)

    # 3. Инференс модели
    input_tensor = torch.from_numpy(np.transpose(img_noisy.astype(np.float32) / 255.0, [2, 0, 1])).unsqueeze(0).to(
        device)

    with torch.no_grad():
        # Прогрев GPU для первого кадра (чтобы не портить статистику времени)
        if fname == files[0]:
            _ = model(input_tensor)

        start_time = time.time()
        output_tensor = model(input_tensor)
        infer_time = (time.time() - start_time) * 1000  # мс

    # 4. Пост-процессинг выхода
    out_np = output_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    img_restored = out_np

    # 5. Расчет метрик
    psnr_val = calculate_psnr(img_gt, img_restored)
    ssim_val = calculate_ssim(img_gt, img_restored)
    de_val = calculate_delta_e(img_gt, img_restored)

    metrics['psnr'].append(psnr_val)
    metrics['ssim'].append(ssim_val)
    metrics['delta_e'].append(de_val)
    metrics['time'].append(infer_time)

    print(f"{fname}:")
    print(f"  PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | ΔE: {de_val:.2f} | Time: {infer_time:.2f}ms")

    # 6. Сохранение картинки сравнения
    create_comparison_grid(img_gt, img_noisy, img_restored, fname)

# Итоговая таблица
print("\n" + "=" * 60)
print("FINAL RESULTS (AVERAGE)")
print("=" * 60)
print(f"Dataset: Set14 ({len(files)} images)")
print(f"Degradation: Gaussian Noise (σ={NOISE_STD})")
print("-" * 60)
print(f"Average PSNR:  {np.mean(metrics['psnr']):.2f} dB")
print(f"Average SSIM:  {np.mean(metrics['ssim']):.4f}")
print(f"Average ΔE:    {np.mean(metrics['delta_e']):.2f}")
print(f"Avg Inference: {np.mean(metrics['time']):.2f} ms")
print("=" * 60)
print(f"Comparison images saved to: {os.path.abspath(SAVE_DIR)}")