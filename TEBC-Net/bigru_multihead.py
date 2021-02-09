import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        # print('Q是不是tensor:')
        # print(torch.is_tensor(Q))

        q_heads = self.WQ(Q).view(batch_size, Q.size(1), self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, K.size(1), self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, V.size(1), self.n_heads, self.d_v).transpose(1, 2)
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)
        # print('output是不是tensor:')
        # print(torch.is_tensor(output))

        return output, attn_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output


# 一个encoder从输入到输出全过程
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model*2, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.linear1 = nn.Linear(d_model * 2,d_ff)
        self.batchnorm1d = nn.BatchNorm1d(d_ff)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model * 2)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        attn_outputs, attn_weights = self.mha(inputs,inputs, inputs, attn_mask)
        attn_outputs = attn_outputs.view(-1,attn_outputs.size(2))

        ffn_outputs = self.linear1(attn_outputs)
        ffn_outputs = self.batchnorm1d(ffn_outputs)
        ffn_outputs = self.tanh(ffn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.linear2(ffn_outputs)
        ffn_outputs = attn_outputs + ffn_outputs

        return ffn_outputs, attn_weights

# 整个encoder的过程，包含多个encoder
class bigru_multihead(nn.Module):
    def __init__(self, config, d_model=300, n_layers=1, n_heads=12, d_ff=2048, pad_id=0):
        super(bigru_multihead, self).__init__()
        self.config = config
        p_drop = config.rnn_fc_dropout
        target_size = config.label
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(config.words_num + 1, d_model)  # (seq_len+1, d_model)

        # layers
        self.embedding = nn.Embedding(config.words_num, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model*2, target_size)
        self.gru = nn.GRU(input_size=config.words_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layer,
                            dropout=config.rnn_dropout,
                            bidirectional=True)

    def forward(self, inputs):

        # self.config = config
        inputs = torch.arange(0, inputs.text.size(0), step=1, device='cpu').repeat(inputs.text.size(1), 1) + 1
        # |inputs| : (batch_size, seq_len)
        outputs = self.embedding(inputs)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        outputs, ht = self.gru(outputs)


        # attention_weights = []
        for layer in self.layers:

            outputs, attn_weights = layer(outputs, attn_pad_mask)



        outputs = self.linear(outputs)
        outputs = outputs.view(inputs.size(0), inputs.size(1), -1)
        scores = torch.mean(outputs, dim=1)

        return scores

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)