import math
import torch
from torch import nn

class PositionWiseEmbedding(nn.Module):

    def __init__(self, input_dim, hid_dim, max_len, dropout_p):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.max_len = max_len
        self.dropout_p = dropout_p

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, inputs):

        # inputs = [batch size, inputs len]
        batch_size = inputs.shape[0]
        inputs_len = inputs.shape[1]

        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(inputs.device)
        scale = self.scale.to(inputs.device)
        embedded = (self.tok_embedding(inputs) * scale) + self.pos_embedding(pos)

        # output = [batch size, inputs len, hid dim]
        output = self.dropout(embedded)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, hid_dim, n_heads):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        assert hid_dim % n_heads == 0

        self.key_weight = nn.Linear(hid_dim, hid_dim)
        self.query_weight = nn.Linear(hid_dim, hid_dim)
        self.value_weight = nn.Linear(hid_dim, hid_dim)
        self.linear_weight = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        # key/query/value (proj) = [batch size, input len, hid dim]
        key_proj = self.key_weight(key)
        query_proj = self.query_weight(query)
        value_proj = self.value_weight(value)

        # compute the weights between query and key
        query_proj = query_proj.view(batch_size, query_len, self.n_heads, self.head_dim)
        query_proj = query_proj.permute(0, 2, 1, 3)
        key_proj = key_proj.view(batch_size, key_len, self.n_heads, self.head_dim)
        key_proj = key_proj.permute(0, 2, 3, 1)

        # energy, attention = [batch size, num heads, query len, key len]
        energy = torch.matmul(query_proj, key_proj) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # output = [batch size, num heads, query len, head dim]
        value_proj = value_proj.view(batch_size, key_len, self.n_heads, self.head_dim)
        value_proj = value_proj.permute(0, 2, 1, 3)
        output = torch.matmul(attention, value_proj)

        # linaer = [batch size, query len, hid dim]
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.hid_dim)
        linear_proj = self.linear_weight(output)
        return linear_proj, attention


class PositionWiseFeedForward(nn.Module):

    def __init__(self, hid_dim, pf_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, inputs):
        # inputs = [batch size, src len, hid dim]
        fc1_output = torch.relu(self.fc1(inputs))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout_p):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_p = dropout_p


        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.position_ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads)
        self.position_ff = PositionWiseFeedForward(hid_dim, pf_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len] 
        self_attention_output, _ = self.self_attention(src, src, src, src_mask)

        # residual connection and layer norm
        self_attention_output = self.dropout(self_attention_output)
        self_attention_output = self.self_attention_layer_norm(src + self_attention_output)

        position_ff_output = self.position_ff(self_attention_output)

        # residual connection and layer norm
        # [batch size, src len, hid dim]
        position_ff_output = self.dropout(position_ff_output)
        output = self.position_ff_layer_norm(self_attention_output + position_ff_output)        
        return output

class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, max_len, dropout_p, n_heads, pf_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.n_layers = n_layers

        self.pos_wise_embedding = PositionWiseEmbedding(input_dim, hid_dim, max_len, dropout_p)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout_p)
            for _ in range(n_layers)
        ])

    def forward(self, src, src_mask = None):

        src = self.pos_wise_embedding(src)
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        for layer in self.layers:
            src = layer(src, src_mask)

        # [batch size, src len, hid dim]
        return src

class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout_p):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_p = dropout_p

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.position_ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads)
        self.position_ff = PositionWiseFeedForward(hid_dim, pf_dim)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, trg, encoded_src, trg_mask, src_mask):
        # encoded_src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len] 
        self_attention_output, _ = self.self_attention(trg, trg, trg, trg_mask)

        # residual connection and layer norm
        self_attention_output = self.dropout(self_attention_output)
        self_attention_output = self.self_attention_layer_norm(trg + self_attention_output)

        encoder_attention_output, _ = self.encoder_attention(
            self_attention_output,
            encoded_src,
            encoded_src,
            src_mask
        )
        encoder_attention_output = self.dropout(encoder_attention_output)
        encoder_attention_output = self.encoder_attention_layer_norm(trg + encoder_attention_output)

        position_ff_output = self.position_ff(encoder_attention_output)

        # residual connection and layer norm
        # [batch size, src len, hid dim]
        position_ff_output = self.dropout(position_ff_output)
        output = self.position_ff_layer_norm(encoder_attention_output + position_ff_output)        
        return output

class Decoder(nn.Module):

    def __init__(self, output_dim, hid_dim, max_len, dropout_p, n_heads, pf_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.n_layers = n_layers

        self.pos_wise_embedding = PositionWiseEmbedding(output_dim, hid_dim, max_len, dropout_p)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout_p)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, trg, encoded_src, trg_mask = None, src_mask = None):
        trg = self.pos_wise_embedding(trg)
        for layer in self.layers:
            trg = layer(trg, encoded_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output

class Transformer(nn.Module):
    """
    
    References
    ----------
    https://pytorch.org/docs/master/generated/torch.nn.Transformer.html
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        src_pad_idx,
        trg_pad_idx,
        max_length=100,
        hid_dim = 512,
        num_head = 8,
        encoder_num_layers = 6,
        decoder_num_layers = 3,
        feedforward_dim = 512,
        dropout = 0.1
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder_embedding = PositionWiseEmbedding(
            input_dim,
            hid_dim,
            max_length,
            dropout
        )
        self.decoder_embedding = PositionWiseEmbedding(
            output_dim,
            hid_dim,
            max_length,
            dropout
        )

        self.encoder = Encoder(
            input_dim, 
            hid_dim,
            max_length,
            dropout, 
            num_head, 
            feedforward_dim, 
            encoder_num_layers
        )
        
        self.decoder = Decoder(
            output_dim, 
            hid_dim,
            max_length,
            dropout,
            num_head,
            feedforward_dim,
            decoder_num_layers
        )

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return decoder_output

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask)

    def decode(self, trg, encoded_src, trg_mask, src_mask):
        decoder_output = self.decoder(trg, encoded_src, trg_mask, src_mask)
        return decoder_output

    def predict(self, src, trg_bos_idx, trg_eos_idx, max_len = 256):
        # separating out the encoder and decoder allows us to generate the encoded source
        # sentence once and share it throughout the target prediction step
        src_mask  = self.make_src_mask(src)
        with torch.no_grad():
            src_encoded = self.encode(src)
  
        # greedy search
        # sequentially predict the target sequence starting from the init sentence token
        trg_tensor = torch.LongTensor([trg_bos_idx]).unsqueeze(0).to(src.device) 
        for _ in range(max_len):
            trg_mask  = self.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output = self.decode(trg_tensor, src_encoded, trg_mask, src_mask)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            trg_tensor = torch.cat((trg_tensor, next_token), dim=1)
            # Stop if end token is generated
            if next_token == trg_eos_idx:
                break
        return trg_tensor


    def make_src_mask(self, src):
        """
        the padding mask is unsqueezed so it can be correctly broadcasted
        when applying the mask to the attention weights, which is of shape
        [batch size, n heads, seq len, seq len].
        """
        src_pad_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(trg.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

