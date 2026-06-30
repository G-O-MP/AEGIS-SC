"""Defense Decoder: 防御用解码器（与 clean decoder 结构相同但独立训练）"""
import torch.nn as nn
from communication.decoder import create_decoder


class DefenseDecoder(nn.Module):
    """独立的防御解码器，结构同 clean decoder"""

    def __init__(self, decoder_kwargs):
        super().__init__()
        self.decoder = create_decoder(**decoder_kwargs)

    def forward(self, z, snr, model_type='WITT'):
        return self.decoder(z, snr, model_type)

    def load_from_clean(self, clean_decoder_state):
        """从 clean decoder 初始化权重"""
        self.decoder.load_state_dict(clean_decoder_state)
