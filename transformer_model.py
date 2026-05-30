import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, 
                seq_len = 256,
                emb_dim = 512):
        '''
        seq_len: max length of a sequence
        emb_dim: embedding dimension per token
        
        Note: the last torch.stack works because C uses row-major order (last dimension first)
        therefore the arrangement in memory is sin, cos, sin,...
        Also we can use view because torch.stack creates a new torch tensor that is already contiguous
        '''
        super().__init__()
        eve_vals = (torch.arange(0, emb_dim, 2) / emb_dim)[None, :]
        odd_vals = (torch.arange(1, emb_dim, 2) / emb_dim)[None, :]
        
        pos = torch.arange(0, seq_len)[:, None]
        eve_freqs = torch.sin(pos / 10000 ** eve_vals)
        odd_freqs = torch.cos(pos / 10000 ** odd_vals)
        self.register_buffer('positional_encoding', 
                             torch.stack((eve_freqs, odd_freqs), dim=-1).view(seq_len, emb_dim))
        
    
    def forward(self, emb):
        t = emb.shape[1]
        return emb + self.positional_encoding[None, :t, ...]

class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 d_dim=512,
                 emb_dim=512):
        super().__init__()
        
        self.Q = nn.Linear(in_features=emb_dim, out_features=d_dim)
        self.K = nn.Linear(in_features=emb_dim, out_features=d_dim)
        self.V = nn.Linear(in_features=emb_dim, out_features=d_dim)
    
    def forward(self, seq, encoder_outputs):
        Q = self.Q(seq)
        K = self.K(encoder_outputs)
        V = self.V(encoder_outputs)
        
        d_k = torch.tensor(Q.shape[-1], dtype=torch.float32)
        
        logits = torch.einsum('btd, bsd -> bts', Q, K)
        norm_logits = logits / torch.sqrt(d_k)
        attention_weights = torch.softmax(norm_logits, dim=-1)
        attention_vals = torch.einsum('bts, bsd-> btd', attention_weights, V)

        return attention_vals

class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 d_dim=512,
                 emb_dim=512):
        super().__init__()
        
        self.Q = nn.Linear(in_features=emb_dim, out_features=d_dim)
        self.K = nn.Linear(in_features=emb_dim, out_features=d_dim)
        self.V = nn.Linear(in_features=emb_dim, out_features=d_dim)
    
    def forward(self, seq):
        '''
        seq: here the input is of shape [b,t,d] where:
            b: is the batch size
            t: is the sequence length
            d: is the embedding dimension
        '''
        
        Q = self.Q(seq)
        K = self.K(seq)
        V = self.V(seq)
        
        d_k = torch.tensor(Q.shape[-1], dtype=torch.float32)
        
        logits = torch.einsum('btd, bsd -> bts', Q, K)
        norm_logits = logits / torch.sqrt(d_k)
        attention_weights = torch.softmax(norm_logits, dim=-1)
        attention_vals = torch.einsum('bts, bsd-> btd', attention_weights, V)

        return attention_vals
    
class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self,
                 d_dim=512,
                 emb_dim=512,
                 num_blocks=8,
                 self_attention=True):
        super().__init__()
        
        self.self_attention = self_attention
        if self_attention:
            self.attention_blocks = nn.ModuleList([
                SelfAttentionBlock(d_dim=d_dim, emb_dim=emb_dim)
                for _ in range(num_blocks)
            ])
        else:
            self.attention_blocks = nn.ModuleList([
                CrossAttentionBlock(d_dim=d_dim, emb_dim=emb_dim)
                for _ in range(num_blocks)
            ])

        self.multihead = nn.Linear(d_dim*num_blocks, d_dim)
        
    def forward(self, seq, *args):
        outputs = []
        if self.self_attention:
            for block in self.attention_blocks:
                outputs.append(block(seq))
        else:
            for block in self.attention_blocks:
                outputs.append(block(seq, *args))
                
        output = torch.cat(outputs, dim=-1)
        
        return self.multihead(output)

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 d_dim=512,
                 emb_dim=512,
                 num_blocks=8):
        super().__init__()
        self.multihead = MultiHeadedAttentionBlock(
            d_dim=d_dim, emb_dim=emb_dim, num_blocks=num_blocks)
        self.layernorm1 = nn.LayerNorm(d_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_dim, d_dim * 4),
            nn.ReLU(),
            nn.Linear(d_dim * 4, d_dim)
        )

        self.layernorm2 = nn.LayerNorm(d_dim)
    
    def forward(self, seq):
        out1 = self.multihead(seq)
        out2 = self.layernorm1(out1 + seq)
        
        out3 = self.feedforward(out2)
        out4 = self.layernorm2(out3 + out2)
        return out4

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_dim=512,
                 emb_dim=512,
                 num_blocks=8):
        
        super().__init__()
        self.multihead = MultiHeadedAttentionBlock(
            d_dim=d_dim, emb_dim=emb_dim, num_blocks=num_blocks
        )
        self.layernorm1 = nn.LayerNorm(d_dim)
        
        self.multihead_cross_attention = MultiHeadedAttentionBlock(
            d_dim=d_dim, emb_dim=emb_dim, num_blocks=num_blocks, 
            self_attention=False
        )
        self.layernorm2 = nn.LayerNorm(d_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_dim, d_dim * 4),
            nn.ReLU(),
            nn.Linear(d_dim * 4, d_dim)
        )
        self.layernorm3 = nn.LayerNorm(d_dim)
        
        self.final_layer = nn.Sequential(
            nn.Linear(d_dim, vocab_size),
            nn.Softmax()
        )
    
    def forward(self, 
                decoder_outputs,
                encoder_outputs):
        '''
        decoder_outputs: [b,s,d]
        encoder_outputs: [b,t,d]
        '''
        
        out1 = self.multihead(decoder_outputs)
        out2 = self.layernorm1(out1 + decoder_outputs)
        
        out3 = self.multihead_cross_attention(out2, encoder_outputs)
        out4 = self.layernorm2(out3 + out2)
        
        out5 = self.feedforward(out4)
        out6 = self.layernorm3(out5 + out4)

        return self.final_layer(out6)
        
class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_blocks=6):
        super().__init__()

positional_encoder = PositionalEncoding()
dummy_embeddings = torch.randn((5, 256, 512))
positional_encoder(dummy_embeddings)
attention_block = MultiHeadedAttentionBlock()
attention_block(dummy_embeddings)