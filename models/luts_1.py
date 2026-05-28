import torch
import torch.nn as nn
import torch.nn.functional as F

class HDLUT(nn.Module):
    def __init__(self, h_weight, d_weight, L, upscale=1):
        super(HDLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0,1,0,0), 'd': (0,1,0,1)}
        self.avg_factor = 2.
    
        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 0:0+H, 1:1+W]
                else: # ktype == 'd'
                    weight = self.d_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 1:1+H, 1:1+W]

                if self.upscale > 1:
                    tmp = weight[img_a.flatten()*self.L + img_b.flatten()].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                    tmp = torch.permute(tmp, (0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                else:
                    # For upscale=1: process each channel separately
                    # weight shape: [L*L, C] where C is number of output channels (3 for RGB)
                    # Each input channel uses corresponding output channel
                    B, C, H, W = img_a.shape
                    idx = img_a.flatten() * self.L + img_b.flatten()  # [B*C*H*W]
                    tmp_flat = weight[idx]  # [B*C*H*W, C_out]
                    
                    # Select the appropriate output channel for each input channel
                    # Reshape to [B, C, H, W, C_out] then select diagonal
                    tmp = tmp_flat.reshape(B, C, H, W, -1)
                    # Use advanced indexing to select channel i from output for input channel i
                    channel_indices = torch.arange(C, device=img_a.device).view(1, C, 1, 1, 1).expand(B, C, H, W, 1)
                    tmp = tmp.gather(-1, channel_indices).squeeze(-1)  # [B, C, H, W]
                    
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor
        
        
class HDBLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, L, upscale=1):
        super(HDBLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2)}
        self.avg_factor = 3.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='reflect').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]
                else:
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 2:2+W]
                    img_c = img_in[:, :, 2:2+H, 1:1+W]
                    weight = self.b_weight

                if self.upscale > 1:
                    tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                                 ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                    tmp = torch.permute(tmp, (0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                else:
                    # For upscale=1: process each channel separately
                    B, C, H, W = img_a.shape
                    idx = img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                    tmp_flat = weight[idx]
                    tmp = tmp_flat.reshape(B, C, H, W, -1)
                    channel_indices = torch.arange(C, device=img_a.device).view(1, C, 1, 1, 1).expand(B, C, H, W, 1)
                    tmp = tmp.gather(-1, channel_indices).squeeze(-1)
                    
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor

class HDTBLUT(nn.Module):
    def __init__(self, h_weight, d_weight, t_weight, b_weight, L, upscale=1):
        super(HDTBLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.t_weight = t_weight
        self.b_weight = b_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 't': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 3, 0, 3), 'd': (0, 3, 0, 3), 't': (0, 3, 0, 3), 'b': (0, 3, 0, 3)}
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 't', 'b']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='reflect').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                    img_d = img_in[:, :, 0:0+H, 3:3+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]
                    img_d = img_in[:, :, 3:3+H, 3:3+W]
                elif ktype == 't':
                    weight = self.t_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 2:2+H, 1:1+W]
                    img_c = img_in[:, :, 3:3+H, 1:1+W]
                    img_d = img_in[:, :, 3:3+H, 2:2+W]
                else:
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 2:2+W]
                    img_c = img_in[:, :, 1:1+H, 3:3+W]
                    img_d = img_in[:, :, 2:2+H, 3:3+W]

                if self.upscale > 1:
                    tmp = weight[img_a.flatten()*self.L*self.L*self.L + img_b.flatten()*self.L*self.L + img_c.flatten()*self.L + img_d.flatten()
                                 ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                    tmp = torch.permute(tmp, (0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                else:
                    # For upscale=1: process each channel separately
                    B, C, H, W = img_a.shape
                    idx = img_a.flatten()*self.L*self.L*self.L + img_b.flatten()*self.L*self.L + img_c.flatten()*self.L + img_d.flatten()
                    tmp_flat = weight[idx]
                    tmp = tmp_flat.reshape(B, C, H, W, -1)
                    channel_indices = torch.arange(C, device=img_a.device).view(1, C, 1, 1, 1).expand(B, C, H, W, 1)
                    tmp = tmp.gather(-1, channel_indices).squeeze(-1)
                    
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor
