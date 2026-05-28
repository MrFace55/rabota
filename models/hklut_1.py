import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bit_plane_slicing, decode_bit_mask
from .luts_1 import HDLUT, HDBLUT, HDTBLUT


class HKLUT(nn.Module): 
    def __init__(self, msb_weights, lsb_weights, msb='hdb', lsb='hd', upscale=1):
        super(HKLUT, self).__init__()
        self.upscale = upscale
        self.bit_mask = '11110000'
        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(self.bit_mask)

        # MSB
        if msb=='hd':
            msb_lut = HDLUT 
        elif msb=='hdb': 
            msb_lut = HDBLUT
        else:
            msb_lut = HDTBLUT

        self.msb_lut = msb_lut(*msb_weights, 2**self.msb_bits, upscale=upscale)


        # LSB
        if lsb=='hd':
            lsb_lut = HDLUT 
        elif lsb=='hdb': 
            lsb_lut = HDBLUT
        else:
            lsb_lut = HDTBLUT        

        self.lsb_lut = lsb_lut(*lsb_weights, 2**self.lsb_bits, upscale=upscale)


    def forward(self, img_lr):

        img_lr_255 = torch.floor(img_lr*255)
        img_lr_msb, img_lr_lsb = bit_plane_slicing(img_lr_255, self.bit_mask)

        # msb
        img_lr_msb = torch.floor_divide(img_lr_msb, self.msb_step)
        MSB_out = self.msb_lut(img_lr_msb)/255.


        # lsb
        img_lr_lsb = torch.floor_divide(img_lr_lsb, self.lsb_step)

        LSB_out = self.lsb_lut(img_lr_lsb)/255.

        if self.upscale > 1:
            img_out = MSB_out + LSB_out + nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr)
        else:
            img_out = MSB_out + LSB_out + img_lr  # residual connection for upscale=1 (color correction)
        
        return torch.clamp(img_out, 0, 1)

