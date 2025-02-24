import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # Create multiple Conv2d layers of kernel_size = 1,3,5,... => (2*i + 1)
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i  # Keeps the output roughly the same spatial size
                )
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass:
        1) Pass x through each Conv2d in self.kernels.
        2) torch.stack() along new dim => shape (B, out_channels, H, W, num_kernels).
        3) .mean(-1) to average across parallel outputs.
        """
        res_list = [conv(x) for conv in self.kernels]
        # Stack along dim=-1 then average along that dimension
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    x = x.float()
    xf = torch.fft.rfft(x, dim=1)
    freq_list = xf.abs().mean(dim=0).mean(dim=-1)
    freq_list[0] = 0  # exclude DC component
    _, top_list = torch.topk(freq_list, k, largest=True)
    top_list = top_list.cpu().numpy()
    period = x.shape[1] // top_list
    return period, xf.abs().mean(dim=-1)[:, top_list]

class TimesBlock(nn.Module):
    """
    Splits input sequence according to multiple periods found by FFT_for_Period,
    then applies the (Inception) convolution block for each period.
    Finally stacks & merges those results with softmax-based weighting.
    """
    def __init__(self, seq_len=128, d_model=128, d_ff=512, num_kernels=6, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, D = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        outputs = []
        for i in range(self.k):
            p = period_list[i]
            length = (((T)//p)+1)*p if T % p != 0 else T
            pad_len = length - T

            out = (
                torch.cat([x, torch.zeros([B, pad_len, D], device=x.device)], dim=1)
                if pad_len > 0 else x
            )
            out = out.reshape(B, length//p, p, D).permute(0, 3, 1, 2)
            out = self.conv(out).permute(0, 2, 3, 1).reshape(B, -1, D)
            outputs.append(out[:, :T, :])

        res = torch.stack(outputs, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, D, 1)
        return (res * period_weight).sum(-1) + x

class FeedForward(nn.Module):
    """Simple 2-layer feedforward network used in Transformers."""
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultivariateAttention(nn.Module):
    """Wrapper around nn.MultiheadAttention with batch_first=True."""
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out

def make_transformer_layers(embed_dim, num_heads, num_layers, dropout=0.2):
    """Builds repeated layers of (Attention -> Norm -> FF -> Norm)."""
    attn_layers = nn.ModuleList([
        MultivariateAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)
    ])
    ff_layers = nn.ModuleList([
        FeedForward(embed_dim, embed_dim*4, dropout) for _ in range(num_layers)
    ])
    norms1 = nn.ModuleList([
        nn.LayerNorm(embed_dim) for _ in range(num_layers)
    ])
    norms2 = nn.ModuleList([
        nn.LayerNorm(embed_dim) for _ in range(num_layers)
    ])
    return attn_layers, ff_layers, norms1, norms2


class TFBlender(nn.Module):
    def __init__(
        self, 
        price_dim, 
        group_dim, 
        condition_dim,
        seq_len=128, 
        embed_dim=128, 
        num_heads=2, 
        num_layers=6,
        dropout=0.2, 
        pos_weight_value=1.0,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.price_dim = price_dim
        self.group_dim = group_dim
        self.condition_dim = condition_dim
        self.N_total = price_dim + group_dim + condition_dim

        # 1) TimesBlock (time-path feature extraction)
        self.times_block = TimesBlock(seq_len, embed_dim, embed_dim*4, 6, 3)
        self.pre_linear_time = nn.Linear(self.N_total, embed_dim)

        # 2) Time-Path Transformer
        (self.time_attn_layers,
         self.time_ff_layers,
         self.time_norms1,
         self.time_norms2) = make_transformer_layers(embed_dim, num_heads, num_layers, dropout)

        # 3) Feature-Path Transformer
        (self.var_attn_layers,
         self.var_ff_layers,
         self.var_norms1,
         self.var_norms2) = make_transformer_layers(seq_len, num_heads, num_layers, dropout)

        # Final linear
        self.final_linear = nn.Linear(embed_dim + self.N_total, 1)

        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32).to(self.device)
        )

    def forward(self, price_data, group_data, x_condition):
        B, T, _ = price_data.shape
        # Expand condition data if needed
        if x_condition.dim() == 2:
            x_condition = x_condition.unsqueeze(1).expand(-1, T, -1)

        # Combine all features
        combined_data = torch.cat([price_data, group_data, x_condition], dim=2).to(self.device)

        # Time path
        time_x = self.times_block(self.pre_linear_time(combined_data))
        for attn, ff, n1, n2 in zip(self.time_attn_layers, self.time_ff_layers, self.time_norms1, self.time_norms2):
            out = attn(time_x)
            time_x = n1(time_x + out)
            out = ff(time_x)
            time_x = n2(time_x + out)

        # Feature path
        var_x = combined_data.transpose(1, 2)
        for attn, ff, n1, n2 in zip(self.var_attn_layers, self.var_ff_layers, self.var_norms1, self.var_norms2):
            out = attn(var_x)
            var_x = n1(var_x + out)
            out = ff(var_x)
            var_x = n2(var_x + out)

        # Final concatenation
        last_time_x = time_x[:, -1, :]
        last_var_x  = var_x[:, :, -1]
        logits = self.final_linear(torch.cat([last_time_x, last_var_x], dim=1))
        return logits

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

