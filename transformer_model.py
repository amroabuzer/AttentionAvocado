import torch.nn as nn
import torch

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
    
    def forward(self, seq,
                encoder_outputs):
        '''
        
        '''
        
        # WE NEED TO MASK OUR MULTIHEAD OUTPUTS HERE!
        out1 = self.multihead(seq)
        out2 = self.layernorm1(out1 + seq)
        
        out3 = self.multihead_cross_attention(out2, encoder_outputs)
        out4 = self.layernorm2(out3 + out2)
        
        out5 = self.feedforward(out4)
        out6 = self.layernorm3(out5 + out4)

        return self.final_layer(out6)
        
class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_blocks=6):
        super().__init__()
        

dummy_embeddings = torch.randn((5, 256, 512))
attention_block = MultiHeadedAttentionBlock()
attention_block(dummy_embeddings)