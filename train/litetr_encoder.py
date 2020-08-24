import sys
sys.path.append('..')
import torch.nn as nn
import torch
from modules import PositionalEncoding, EncoderLayer



class LiteTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, num_encoder_layers,label_vocab_size, dropout=0.1):
        super(LiteTransformerEncoder, self).__init__()

        self.padding = 1
        self.kernel_size = 3
        self.stride = 2
        self.dilation = 1

        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=1,
                                                 out_channels=d_model//2,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model//2),
                                       nn.ReLU(inplace=True),

                                       nn.Conv1d(in_channels=d_model//2,
                                                 out_channels=d_model,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=d_model),
                                       nn.ReLU(inplace=True),
                                       )
        # TODO: why padding_idx=0
        self.position_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout)

        self.stack_layers = nn.ModuleList(
            [EncoderLayer(index=i, d_model=d_model, d_ff=d_ff, n_head=n_head, dropout=dropout) for i in range(
                num_encoder_layers)])   #need change
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.final_proj = nn.Linear(d_model, label_vocab_size)

    def forward(self, signal, signal_lengths):
        """
        :param signal: a tensor shape of [batch, length, 1]
        :param signal_lengths: a tensor shape of [batch,]
        :return:
        """

        max_len = signal.size(1)

        max_len = int(((max_len + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) / self.stride + 1))

        max_len = int(((max_len + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) / self.stride + 1))

        new_signal_lengths = ((signal_lengths + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) / self.stride + 1).int()

        new_signal_lengths = ((new_signal_lengths + 2 * self.padding - self.dilation * (
                self.kernel_size - 1) - 1) / self.stride + 1).int()

        src_mask = torch.tensor([[0] * v.item() + [1] * (max_len - v.item()) for v in new_signal_lengths],
                                dtype=torch.uint8).unsqueeze(-2).to(signal.device) #[N,1,L]need change

        signal = signal.transpose(-1, -2)  # (N,C,L)


        embed_out = self.src_embed(signal)  # (N,C,L)

        embed_out = embed_out.transpose(-1, -2)  # (N,L,C)
        enc_output = self.position_encoding(embed_out)

        for layer in self.stack_layers:
            enc_output, enc_slf_attn = layer(enc_output, src_mask)

        enc_output = self.layer_norm(enc_output)

        enc_output = self.final_proj(enc_output)

        return enc_output, new_signal_lengths
