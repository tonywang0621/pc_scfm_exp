# ==============================================================================
# IMPORTANT NOTICE:
# This code is exclusively for the BioASP lab version and should not be distributed outside the lab.
# Unauthorized distribution or sharing of this code can result in severe penalties
# and is strictly prohibited.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
import sys
sys.path.append('mamba')
from functools import partial
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
        
def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    # Version 1
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)

    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def mag_pha_stft_loss(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    real_part = stft_spec.real
    imag_part = stft_spec.imag
    stft_spec = torch.stack((real_part, imag_part), dim=-1)
    # Version 1
    #mag = torch.abs(stft_spec)
    #pha = torch.angle(stft_spec)
    # Version 2 
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:,:,:,1] + (1e-10), stft_spec[:,:,:,0] + (1e-5))

    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav

class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4, use_fast_path=True),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4, use_fast_path=True),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        
        return residual
    

class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))
        depth = h.get('edepth',4)
        self.dense_block = DenseBlock(h, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        depth = h.get('mdepth',4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        depth = h.get('pdepth',4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

class ComplexDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(ComplexDecoder, self).__init__()
        depth = h.get('pdepth',4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.cat((x_r, x_i), dim=1)
        return x


class TSMambaBlock(nn.Module):
    def __init__(self, h):
        super(TSMambaBlock, self).__init__()
        self.h = h

        #self.time_mamba = Mamba(d_model=h.dense_channel,  d_state=16, d_conv=4, expand=2)
        #self.freq_mamba = Mamba(d_model=h.dense_channel,  d_state=16, d_conv=4, expand=2)

        self.time_mamba = MambaBlock(in_channels=h.dense_channel, n_layer=1, bidirectional=True)
        self.freq_mamba = MambaBlock(in_channels=h.dense_channel, n_layer=1, bidirectional=True)

        self.tlinear = nn.ConvTranspose1d(
            h.dense_channel * 2, h.dense_channel, 1, stride=1
        )
        if self.h.get('fmamba',True):
            self.flinear = nn.ConvTranspose1d(
                h.dense_channel * 2, h.dense_channel, 1, stride=1
            )

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        #x = self.time_conformer(x) + x
        #print(x.shape)
        #print(self.time_mamba(x).shape)
        #print(self.tlinear( self.time_mamba(x) ).shape)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c)
        #print(x.shape)
        if self.h.get('fmamba',True):
            x = x.permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
            x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
            
            #x = self.freq_conformer(x) + x
            x = x.view(b, t, f, c).permute(0, 2, 1, 3)
        x = x.permute(0, 3, 2, 1)
        #fesfesf
        return x

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MECGE(nn.Module):
    def __init__(self, config):
        super(MECGE, self).__init__()

        h = AttrDict(config['model'])
        self.fea =  h.get('fea','pha')
        self.h = h
        self.norm = h.norm
        self.loss_fn = h.loss_fn.split('+')
        self.num_tscblocks = h.num_tscblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSMamba = nn.ModuleList([])
        for i in range(h.num_tscblocks):
            self.TSMamba.append(TSMambaBlock(h))
        
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        if self.fea=='cpx':
            self.complex_decoder = ComplexDecoder(h, out_channel=1)
        elif self.fea=='wav':
            self.encoder = nn.Conv1d(1, (self.h.n_fft//2+1)*2, self.h.win_size, self.h.hop_size, padding=self.h.win_size//2)
            self.decoder = nn.ConvTranspose1d((self.h.n_fft//2+1)*2, 1, self.h.win_size, self.h.hop_size, padding=self.h.win_size//2, output_padding=0)
            self.complex_decoder = ComplexDecoder(h, out_channel=1)
        elif self.fea=='pha':
            self.phase_decoder = PhaseDecoder(h, out_channel=1)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

    @torch.no_grad()
    def denoising(self, noisy_audio):

        if self.norm=='1':
            norm_factor = torch.sqrt(noisy_audio.shape[-1] / torch.sum(noisy_audio ** 2.0, -1, keepdim=True))
        elif self.norm=='2':
            norm_factor = 1 / noisy_audio.abs().max(-1, keepdim=True)[0]
        else:
            norm_factor = torch.ones((noisy_audio.shape[0],1,1),device=noisy_audio.device)
            
        noisy_audio = noisy_audio * norm_factor
        noisy_audio = noisy_audio.squeeze(1)
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        
        if self.fea=='cpx':
            x = noisy_com.permute(0, 3, 2, 1) # [B, 2, T, F]
        elif self.fea=='pha':
            noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
            x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        elif self.fea=='wav':
            x = self.encoder(noisy_audio.unsqueeze(1))
            B, C, T = x.shape
            x = x.view(B, 2, -1, T).permute(0, 1, 3, 2)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        
        x = self.dense_encoder(x)
        for i in range(self.num_tscblocks):
            x = self.TSMamba[i](x)
        
        mag_g = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)

        if self.fea=='cpx':
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack((mag_g*torch.cos(noisy_pha),
                                    mag_g*torch.sin(noisy_pha)), dim=-1)
            com_g = com_g + com_d
            pha_g = torch.angle(torch.complex(com_g[...,0], com_g[...,1]))
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='pha':
            pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='wav':
            com_d = self.complex_decoder(x).permute(0, 1, 3, 2).reshape(B, C, T)
            audio_g = self.decoder(com_d).squeeze(1)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        audio_g = audio_g.unsqueeze(1)
        audio_g = audio_g/norm_factor

        return audio_g

    def forward(self, clean_audio, noisy_audio): # [B, F, T]

        if self.norm=='1':
            norm_factor = torch.sqrt(noisy_audio.shape[-1] / torch.sum(noisy_audio ** 2.0, -1, keepdim=True))
        elif self.norm=='2':
            norm_factor = 1 / noisy_audio.abs().max(-1, keepdim=True)[0]
        else:
            norm_factor = torch.ones((noisy_audio.shape[0],1,1),device=noisy_audio.device)

        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = (noisy_audio * norm_factor).squeeze(1)
        
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 

        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]

        if self.fea=='cpx':
            x = noisy_com.permute(0, 3, 2, 1) # [B, 2, T, F]
        elif self.fea=='pha':
            noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
            x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        elif self.fea=='wav':
            x = self.encoder(noisy_audio.unsqueeze(1))
            B, C, T = x.shape
            x = x.view(B, 2, -1, T).permute(0, 1, 3, 2)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSMamba[i](x)
        
        mag_g = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)

        if self.fea=='cpx':
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack((mag_g*torch.cos(noisy_pha),
                                    mag_g*torch.sin(noisy_pha)), dim=-1)
            com_g = com_g + com_d
            # mag_g = torch.abs(torch.complex(com_g[...,0], com_g[...,1]))
            pha_g = torch.angle(torch.complex(com_g[...,0], com_g[...,1]))
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='pha':
            pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
            com_g = torch.stack((mag_g*torch.cos(pha_g),
                                        mag_g*torch.sin(pha_g)), dim=-1)
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='wav':
            com_d = self.complex_decoder(x).permute(0, 1, 3, 2).reshape(B, C, T)
            audio_g = self.decoder(com_d).squeeze(1)
            _, _, com_g = mag_pha_stft_loss(audio_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        loss_gen_all = 0

        # Time Loss
        if 'time' in self.loss_fn:
            loss_time = F.l1_loss(clean_audio, audio_g, reduction='none')
            loss_time = (loss_time/norm_factor.squeeze(-1)).mean()
            loss_gen_all += loss_time * 0.5

        # L2 Complex Loss
        if 'com' in self.loss_fn:
            loss_com = F.mse_loss(clean_com, com_g, reduction='none') * 2
            loss_com = (loss_com/norm_factor.unsqueeze(-1)).mean()
            loss_gen_all += loss_com * 0.5

        # Consistancy Loss
        if 'con' in self.loss_fn:
            _, _, com_con = mag_pha_stft_loss(audio_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
            loss_con = F.mse_loss(com_g, com_con, reduction='none') * 2
            loss_con = (loss_con/norm_factor.unsqueeze(-1)).mean()
            loss_gen_all += loss_con * 0.5
        
        return loss_gen_all


