
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
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        residual = Q
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
        output = self.dropout(output)
        output += residual

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

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output

#一个encoder从输入到输出全过程
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        # print('输入时：')
        # print(inputs.size(0))
        # print(inputs.size(1))
        # print(inputs.size(2))

        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)

        # print('ffn_outputs是不是tensor:')
        # print(torch.is_tensor(ffn_outputs))
        # print('经过一层encoder输入后：')
        # print(ffn_outputs.size(0))
        # print(ffn_outputs.size(1))
        # print(ffn_outputs.size(2))

        return ffn_outputs, attn_weights

#整个encoder的过程，包含多个encoder
class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len    (int)    : input sequence length
        d_model    (int)    : number of expected features in the input
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
        pad_id     (int)    : pad token id
    Examples:
    # >>> encoder = TransformerEncoder(vocab_size=1000, seq_len=512)
    # >>> inp = torch.arange(512).repeat(2, 1)
    # >>> encoder(inp)
    """

    def __init__(self, config,d_model=300, n_layers=1, n_heads=1, p_drop=0.1, d_ff=2048, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.config = config
        target_size = config.label
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(config.words_num+1, d_model)  # (seq_len+1, d_model)

        # layers
        self.embedding = nn.Embedding(config.words_num,d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        # self.layers = EncoderLayer(d_model, n_heads, p_drop, d_ff)
        # layers to classify
        self.linear = nn.Linear(d_model,target_size)
        self.softmax = nn.Softmax(dim=-1)

        self.mha = MultiHeadAttention(config.hidden_size, 1)

        self.linear1 = nn.Linear(config.hidden_size, 2048)
        self.batchnorm1d = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.linear2 = nn.Linear(2048, target_size)

        self.position_enc = PositionalEncoding(300, 300)



    def forward(self,inputs):
        # x = inputs
        # text = inputs.text
        # inputs = self.embedding(text)
        # # self.config = config
        # inputs = inputs.transpose(0,1)
        inputs = torch.arange(0,inputs.text.size(0), step =1,device='cpu').repeat(inputs.text.size(1), 1)
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(0,inputs.size(1), step =1,device='cpu').repeat(inputs.size(0), 1)
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)
        # positions = positions.view(config.batch_size,config.words_dim,-1)

        outputs = self.embedding(inputs) + self.dropout(self.position_enc(self.embedding(positions)))
                  #+ self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)
        # print('outputs是不是tensor:')
        # print(torch.is_tensor(outputs))#true

        # attention_weights = []
        # for layer in self.layers:
        #
        #     outputs, attention_weights = layer(outputs, attn_pad_mask)

            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            # attention_weights.append(attn_weights)

        outputs,weight = self.mha(outputs,outputs,outputs,attn_pad_mask)
        outputs = outputs.view(-1, outputs.size(2))
        outputs = self.linear1(outputs)
        outputs = self.batchnorm1d(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        tags = outputs.view(inputs.size(1), inputs.size(0), -1)
        scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)


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



class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()