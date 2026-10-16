"""
FT (Force/Torque) 编码器。

输入是按时间展平的 FT 序列，每个时间步包含 6 维力/力矩值。
典型输入形状为 `(B, input_dim)` 或 `(input_dim,)`，其中 `input_dim` 需要能被 6 整除。
时序编码器会在内部将其重排为 `(B, T, 6)`，其中 `T = input_dim // 6`。

支持的编码器:
- MLP: 直接对展平后的 FT 向量编码
- LSTM: 重排为时序后用 LSTM 编码
- TCN: 重排为时序后用 TCN 编码
- Causal Conv: 重排为时序后用因果卷积编码
"""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CausalConv1D(nn.Module):
    """因果1D卷积：只在左侧填充以保证因果性"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,stride: int =1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0, stride=stride)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)  # (B, T, C)


class TemporalBlock(nn.Module):
    """TCN的基础模块：两层因果卷积 + LayerNorm + ReLU + Dropout + 残差连接"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1D(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1D(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
            nn.init.kaiming_uniform_(self.downsample.weight, nonlinearity='relu')
            nn.init.zeros_(self.downsample.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(F.relu(self.norm1(self.conv1(x))))
        out = self.dropout(F.relu(self.norm2(self.conv2(out))))
        if self.downsample:
            residual = self.downsample(residual.transpose(1, 2)).transpose(1, 2)
        return F.relu(out + residual)


class FTMLPEncoder(nn.Module):
    """
    MLP 编码器。

    输入: `(B, input_dim)` 或 `(input_dim,)`
    输出: `(B, output_dim)` 或 `(output_dim,)`
    """
    def __init__(self, input_dim: int = 600, hidden_dims: Tuple[int, ...] = (256, 128), output_dim: int = 128):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        self.output_dim = output_dim
        
        # Kaiming初始化 (适用于ReLU)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.output_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = self.mlp(x)
        x = self.output_proj(x)
        return x.squeeze(0) if squeeze else x


class FTLSTMEncoder(nn.Module):
    """
    LSTM 编码器。

    输入会自动重排为 `(B, T, 6)`，其中 `T = input_dim // 6`。
    输出: `(B, output_dim)` 或 `(output_dim,)`
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
        nn.init.kaiming_uniform_(self.output_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        
        # (B, input_dim) -> (B, T, 6)
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)
        
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)
        x = self.output_proj(lstm_out[:, -1, :])  # 取最后一个时间步
        
        return x.squeeze(0) if squeeze else x


class FTTCNEncoder(nn.Module):
    """
    TCN 编码器。

    输入会自动重排为 `(B, T, 6)`，其中 `T = input_dim // 6`。
    输出: `(B, output_dim)` 或 `(output_dim,)`
    """
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 64,
        num_channels: Sequence[int] = (32, 64, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        pooling: str = 'last'  # 'last', 'mean', 'max'

    ):
        super().__init__()
        self.input_dim = input_dim
        self.pooling = pooling
        self.output_dim = output_dim
        
        # 构建TCN blocks
        self.blocks = nn.ModuleList()
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            self.blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        
        self.output_proj = nn.Linear(num_channels[-1], output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        nn.init.kaiming_uniform_(self.output_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        
        # (B, input_dim) -> (B, T, 6)
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)
        
        for block in self.blocks:
            x = block(x)
        
        # Pooling
        if self.pooling == 'last':
            x = x[:, -1, :]
        elif self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        
        x = self.output_norm(self.output_proj(x))
        
        return x.squeeze(0) if squeeze else x


class SequenceCausalConv1D(nn.Module):
    """Channel-first causal conv for sequence modules that operate on [B, C, T]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class SequenceTemporalBlock(nn.Module):
    """TCN residual block used by force sequence encoders/decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = SequenceCausalConv1D(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = SequenceCausalConv1D(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return F.gelu(y + residual)


class FTSequenceTCNEncoder(nn.Module):
    """
    TCN encoder for non-flattened force sequences.

    Input: [B, T, force_dim]
    Output: [B, embed_dim]
    """

    def __init__(
        self,
        force_dim: int,
        hidden_channels: Tuple[int, ...],
        embed_dim: int,
        kernel_size: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling)
        blocks = []
        in_ch = int(force_dim)
        for i, out_ch in enumerate(hidden_channels):
            blocks.append(
                SequenceTemporalBlock(in_ch, int(out_ch), kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = int(out_ch)
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(in_ch, embed_dim), nn.LayerNorm(embed_dim))
        self.output_dim = int(embed_dim)

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        if force.ndim != 3:
            raise ValueError(f"Expected force [B,T,C], got {tuple(force.shape)}")
        x = force.transpose(1, 2).contiguous()
        x = self.blocks(x).transpose(1, 2).contiguous()
        if self.pooling == "last":
            pooled = x[:, -1]
        elif self.pooling == "mean":
            pooled = x.mean(dim=1)
        elif self.pooling == "max":
            pooled = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling={self.pooling!r}")
        return self.proj(pooled)


class FTSequenceTokenEncoder(nn.Module):
    """
    TCN encoder for non-flattened force sequences that preserves one token per history step.

    Input: [B, T, force_dim]
    Output: [B, T, embed_dim]
    """

    def __init__(
        self,
        force_dim: int,
        embed_dim: int,
        hidden_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        include_delta: bool = True,
        include_force_norm: bool = True,
    ) -> None:
        super().__init__()
        self.force_dim = int(force_dim)
        self.embed_dim = int(embed_dim)
        self.include_delta = bool(include_delta)
        self.include_force_norm = bool(include_force_norm)

        in_ch = self.force_dim
        if self.include_delta:
            in_ch += self.force_dim
        if self.include_force_norm:
            in_ch += 1

        blocks = []
        current_ch = in_ch
        for i, out_ch in enumerate(hidden_channels):
            blocks.append(
                SequenceTemporalBlock(
                    current_ch,
                    int(out_ch),
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            current_ch = int(out_ch)
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(current_ch, self.embed_dim), nn.LayerNorm(self.embed_dim))
        self.output_dim = self.embed_dim

    def _build_features(self, force: torch.Tensor) -> torch.Tensor:
        features = [force]
        if self.include_delta:
            delta_force = force[:, 1:] - force[:, :-1]
            delta_force = F.pad(delta_force, (0, 0, 1, 0))
            features.append(delta_force)
        if self.include_force_norm:
            features.append(force.norm(dim=-1, keepdim=True))
        return torch.cat(features, dim=-1)

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        if force.ndim != 3 or force.shape[-1] != self.force_dim:
            raise ValueError(f"Expected force [B,T,{self.force_dim}], got {tuple(force.shape)}")
        x = self._build_features(force)
        x = self.blocks(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return self.proj(x)


class FTSequenceTCNDecoder(nn.Module):
    """
    TCN decoder for force sequence reconstruction.

    Input: [B, embed_dim]
    Output: [B, window, force_dim]
    """

    def __init__(
        self,
        window: int,
        force_dim: int,
        hidden_channels: Tuple[int, ...],
        embed_dim: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.seed = nn.Linear(embed_dim, int(hidden_channels[0]) * self.window)
        blocks = []
        channels = list(hidden_channels)
        in_ch = int(channels[0])
        for i, out_ch in enumerate(channels[1:] + [channels[-1]]):
            blocks.append(
                SequenceTemporalBlock(in_ch, int(out_ch), kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = int(out_ch)
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(in_ch, int(force_dim), kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.seed(z).reshape(z.shape[0], -1, self.window)
        x = self.blocks(x)
        return self.head(x).transpose(1, 2).contiguous()


class FTCausalConvEncoder(nn.Module):
    """
    因果卷积 FT 编码器。

    输入会自动重排为 `(B, T, 6)`，随后经过多层 stride=2 的因果卷积，
    最终取最后一个时间位置的特征作为输出。
    """
    def __init__(self, output_dim):
        """
        Force encoder taken from self-supervised code.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = 6

        self.frc_encoder = nn.Sequential(
        CausalConv1D(6, 32, kernel_size=3, dilation=1, stride=1),
        nn.LeakyReLU(0.1, inplace=True),
        CausalConv1D(32, 64, kernel_size=3, dilation=2, stride=1),
        nn.LeakyReLU(0.1, inplace=True),
        CausalConv1D(64, 64, kernel_size=3, dilation=4, stride=1),
        nn.LeakyReLU(0.1, inplace=True),
        CausalConv1D(64, 128, kernel_size=3, dilation=8, stride=1),
        nn.LeakyReLU(0.1, inplace=True),
        CausalConv1D(128, self.output_dim, kernel_size=3, dilation=16, stride=1),
        nn.LeakyReLU(0.1, inplace=True),
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        
        # (B, input_dim) -> (B, T, 6)
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)
        x =  self.frc_encoder(x)

        x = x[:, -1, :]

        return x.squeeze(0) if squeeze else x

class FTEncoder(nn.Module):
    """
    统一的 FT 编码器接口。

    输入: `(B, input_dim)` 或 `(input_dim,)` 的展平 FT 数据
    输出: `(B, output_dim)` 或 `(output_dim,)` 的编码特征

    Args:
        encoder_type: 'mlp', 'lstm',  'tcn' 或 'causal_conv'
        input_dim: 展平后的 FT 输入维度，需能被 6 整除
        output_dim: 输出特征维度
        **kwargs: 传递给具体编码器的额外参数
    """
    def __init__(
        self,
        encoder_type: str = 'mlp',
        input_dim: int = 600,
        mlp_hidden_dim: Tuple[int, ...] = (256,128),
        lstm_hidden_dim: int = 64,
        tcn_num_channels:Tuple[int, ...] = (32, 64, 64, 128),
        output_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        self.encoder_type = encoder_type
        print(f"\033[92mInitializing FTEncoder with type '{encoder_type}' and output_dim={output_dim}\033[0m")
        
        if encoder_type == 'mlp':
            self.encoder = FTMLPEncoder(input_dim=input_dim, hidden_dims=mlp_hidden_dim, output_dim=output_dim)
        elif encoder_type == 'lstm':
            self.encoder = FTLSTMEncoder(
                input_dim=6,  # 每个时间步的 FT 通道数固定为 6
                hidden_dim=lstm_hidden_dim,
                output_dim=output_dim,
                num_layers=kwargs.get('lstm_num_layers', 2)
            )
        elif encoder_type == 'tcn':
            self.encoder = FTTCNEncoder(
                input_dim=6,
                num_channels=tcn_num_channels,
                output_dim=output_dim,
                kernel_size=kwargs.get('kernel_size', 3),
                dropout=kwargs.get('dropout', 0.1),
                pooling=kwargs.get('pooling', 'last')
            )
        elif encoder_type == 'causal_conv':
            self.encoder = FTCausalConvEncoder(output_dim=output_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'mlp', 'lstm', 'tcn', or 'causal_conv'")
        
        self.output_dim = self.encoder.output_dim
        self.input_dim = input_dim
    
    def forward(self, ft_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ft_data: `(B, input_dim)` 或 `(input_dim,)` 的展平 FT 数据
        Returns:
            `(B, output_dim)` 或 `(output_dim,)` 的编码特征
        """
        if ft_data.dim() == 1:
            ft_data = ft_data[-self.input_dim:]
        else:
            ft_data = ft_data[:, -self.input_dim:]
        return self.encoder(ft_data)


if __name__ == "__main__":
    print("Testing FT Encoders (Simplified PyTorch)")
    print("=" * 50)
    
    batch_size = 4
    ft_steps = 60
    ft_flat_dim = ft_steps * 6
    ft_data_batched = torch.randn(batch_size, ft_flat_dim)
    ft_data_single = torch.randn(ft_flat_dim)
    
    for enc_type in ['mlp', 'lstm', 'tcn', 'causal_conv']:
        print(f"\n{enc_type.upper()} Encoder:")
        encoder = FTEncoder(input_dim=ft_data_batched.shape[1],encoder_type=enc_type, output_dim=64)
        encoder.eval()
        
        out_b = encoder(ft_data_batched)
        out_s = encoder(ft_data_single)
        print(f"  Batched: {ft_data_batched.shape} -> {out_b.shape}")
        print(f"  Single:  {ft_data_single.shape} -> {out_s.shape}")
        print(f"  Output dim: {encoder.output_dim}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
