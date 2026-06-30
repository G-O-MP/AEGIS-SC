"""Dual Decoder: clean decoder + defense decoder 双路结构"""
import torch
import torch.nn as nn
from communication.decoder import create_decoder
from .defense_decoder import DefenseDecoder


class DualDecoder(nn.Module):
    """双解码器架构:
       - clean_decoder: 正常语义重建
       - defense_decoder: 抗塌缩防御重建
    """

    def __init__(self, decoder_kwargs, freeze_clean=True):
        super().__init__()
        self.clean_decoder = create_decoder(**decoder_kwargs)
        self.defense_decoder = DefenseDecoder(decoder_kwargs)

        if freeze_clean:
            for p in self.clean_decoder.parameters():
                p.requires_grad = False

    def forward(self, z, snr, model_type='WITT', mode='clean'):
        """mode: 'clean' → clean decoder; 'defense' → defense decoder; 'both' → (clean, defense)"""
        if mode == 'clean':
            return self.clean_decoder(z, snr, model_type)
        elif mode == 'defense':
            return self.defense_decoder(z, snr, model_type)
        elif mode == 'both':
            return self.clean_decoder(z, snr, model_type), self.defense_decoder(z, snr, model_type)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_clean_state(self, state_dict):
        self.clean_decoder.load_state_dict(state_dict)

    def load_defense_state(self, state_dict):
        self.defense_decoder.decoder.load_state_dict(state_dict)
