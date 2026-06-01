import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
from utils import decode_bit_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sanitize_name(name):
    return name.replace(':', '_').replace('/', '_').replace('\\', '_')

def parse_args():
    parser = argparse.ArgumentParser("Transfer Setting")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--upscale", nargs='+', type=int, default=[1],
                        help="upscaling factors")
    parser.add_argument('--msb', type=str, default='hdt', choices=['hdb', 'hd', 'hdt'])
    parser.add_argument('--lsb', type=str, default='hdt', choices=['hdb', 'hd', 'hdt'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=48, help="number of filters in intermediate layers")
    parser.add_argument('--lut-dir', type=str, default='./luts_converted',
                        help="Output directory for LUT files")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb:{}-lsb:{}-act:{}-nf:{}-{}".format(args.msb, args.lsb, args.act_fn, args.n_filters, factors)

    act_fn_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]
    return args


def get_input_tensor(bits, base_steps, n_pixels=3):
    L = 2 ** bits
    base_step_ind=torch.arange(0, L, 1)
    base=base_steps*base_step_ind/255.0
    index_nD=torch.meshgrid(*[base for _ in range(n_pixels)])
    input_tensor=torch.cat([index_nD[i].flatten().unsqueeze(1) for i in range(len(index_nD))], 1).unsqueeze(1)
    return input_tensor

pixel_dict = {'hdb': 3, 'hd': 2, 'hdt': 4}

if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(f"Loading checkpoint from: {args.checkpoint}")

    models = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    for i, s in enumerate(args.upscale):
        model = HKNet(msb=args.msb, lsb=args.lsb, nf=args.n_filters, upscale=s, act=args.act_fn).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        models.append(model)

    safe_exp_name = sanitize_name(args.exp_name)
    lut_dir = os.path.join(args.lut_dir, safe_exp_name)
    if not os.path.isdir(lut_dir):
        os.makedirs(lut_dir, exist_ok=True)

    msb_bits, lsb_bits, msb_step, lsb_step = decode_bit_mask('11110000')

    with torch.no_grad():
        for stage in range(n_stages):
            model = models[stage]
            model.eval()
            msb_luts = filter(lambda a: 'msb' in a and 'lut' in a , dir(model))
            lsb_luts = filter(lambda a: 'lsb' in a and 'lut' in a , dir(model))

            for lut in msb_luts:
                msb_unit = model.__getattr__(lut)
                lut_input = get_input_tensor(msb_bits, msb_step, n_pixels=pixel_dict[args.msb])
                lut_input = msb_unit.get_lut_input(lut_input).to(device)

                B = max(1, lut_input.size(0) // 100)
                outputs = []

                for b in range(100):
                    start = b * B
                    end = None if b == 99 else (b + 1) * B
                    batch_input = lut_input[start:end]
                    if batch_input.size(0) == 0:
                        continue
                    batch_output = msb_unit(batch_input)
                    results = torch.floor(torch.clamp(batch_output, -1, 1) * 127).cpu().data.numpy().astype(np.int8)
                    outputs.append(results)

                results = np.concatenate(outputs, 0)
                lut_path = os.path.join(lut_dir, f'S{stage}_{lut.upper()}_x{args.upscale[stage]}_{msb_bits}bit_int8.npy')
                np.save(lut_path, results)
                print(f"Resulting LUT size: {results.shape}, Saved to {lut_path}")

            for lut in lsb_luts:
                lsb_unit = model.__getattr__(lut)
                lut_input = get_input_tensor(lsb_bits, lsb_step, n_pixels=pixel_dict[args.lsb])
                lut_input = lsb_unit.get_lut_input(lut_input).to(device)

                B = max(1, lut_input.size(0) // 100)
                outputs = []

                for b in range(100):
                    start = b * B
                    end = None if b == 99 else (b + 1) * B
                    batch_input = lut_input[start:end]
                    if batch_input.size(0) == 0:
                        continue
                    batch_output = lsb_unit(batch_input)
                    results = torch.floor(torch.clamp(batch_output, -1, 1) * 127).cpu().data.numpy().astype(np.int8)
                    outputs.append(results)

                results = np.concatenate(outputs, 0)
                lut_path = os.path.join(lut_dir, f'S{stage}_{lut.upper()}_x{args.upscale[stage]}_{lsb_bits}bit_int8.npy')
                np.save(lut_path, results)
                print(f"Resulting LUT size: {results.shape}, Saved to {lut_path}")

    print(f"\nAll LUTs saved to: {lut_dir}")
