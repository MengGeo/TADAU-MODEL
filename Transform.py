import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import Dropout

# ==========================================
# (batch_size,in_channels,height)
class Embedding(nn.Module):
    def __init__(self, in_channels, height, dropout_rate=0.0):
        super(Embedding, self).__init__()
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, height, in_channels))
        self.dropout = Dropout(dropout_rate)

    def forward(self, xin):
        #xin:(batch_size,in_channels,height)
        xin = xin.transpose(-1, -2)  # (batch_size, height, in_channels)
        embeddings = xin + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

# embed_dim:
# num_heads:
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, dropout_rate=0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim    #
        self.num_heads = num_heads    #
        self.head_dim = embed_dim // num_heads  #

        # Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        #
        self.proj_dropout = Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        #
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  #
        #
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        #
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        #
        output = torch.matmul(attn_probs, v)
        #
        output = output.transpose(dim0=1, dim1=2).contiguous().view(batch_size, seq_length, self.embed_dim)
        #
        output = self.out_proj(output)
        output = self.proj_dropout(output)
        return output

#
# embed_dim:
# ff_dim_n:
class MLP(nn.Module):
    def __init__(self, embed_dim=64, ff_dim_n=4, dropout=0.0):
        super(MLP, self).__init__()
        ff_dim = ff_dim_n * embed_dim  #
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        self.relu = nn.ReLU()

    #
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, xx):
        xx = self.fc1(xx)
        xx = self.relu(xx)
        xx = self.dropout(xx)
        xx = self.fc2(xx)
        return xx

# ==========================================
# embed_dim:
# num_heads:
# ff_dim_n:
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ff_dim_n=4, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = MLP(embed_dim, ff_dim_n, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xi):
        h = xi
        #
        attn_output = self.self_attn(self.norm1(xi))
        #
        xi = h + attn_output
        h = xi
        #
        ff_output = self.feed_forward(self.norm2(xi))
        #
        ff_output = h + ff_output
        return ff_output

# ===================== =====================
# num_layers:Transformer
class Transformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, num_layers=3, ff_dim_n=4,
                 dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim_n, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xin):
        for layer in self.layers:
            xin = layer(xin)
        xin = self.norm(xin)
        return xin

# ===================== =====================
class EmbedTrans(nn.Module):
    def __init__(self, in_channels, height, num_heads=8, num_layers=3,
                 ff_dim_n=2, dropout=0.0):
        super(EmbedTrans, self).__init__()
        embed_dim = in_channels  #
        #
        self.embeddings = Embedding(in_channels, height, dropout)
        # Transformer
        self.Trans = Transformer(embed_dim, num_heads, num_layers, ff_dim_n,
                                 dropout)

    def forward(self, input_seis):  # (batch_size,in_channels,height)
        embedding_output = self.embeddings(input_seis)  # (batch_size, height, in_channels)
        Trans_out = self.Trans(embedding_output)  # (B, height, in_channels)
        Trans_out = Trans_out.transpose(-1, -2)  # (B, in_channels, height)
        return Trans_out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.randn(50, 1, 256, 13)
    # x= x.to(device) #
    MultiHead = MultiHeadAttention(embed_dim=128, num_heads=32)
    # summary(MultiHead, input_size=(50, 256, 128)) #
    EmTr = EmbedTrans(in_channels=128, height=400, num_heads=16, num_layers=5, ff_dim_n=4)
    # summary(EmTr, input_size=(50, 128, 400)) #
    Transform = Transformer(embed_dim=64, num_heads=8, num_layers=3)
    summary(Transform, input_size=(50,256,64))

    total_params = sum(p.numel() for p in Transform.parameters())
    print(f"Total number of parameters: {total_params}")