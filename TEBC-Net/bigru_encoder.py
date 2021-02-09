import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter



class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff,p_drop):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.batchnorm1d = nn.BatchNorm1d(d_model * 2)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model * 2, d_model)


    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)
        outputs = self.linear1(inputs)
        outputs = self.batchnorm1d(outputs)
        outputs = self.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        outputs = outputs.view(inputs.size(0),inputs.size(1),-1)

        output = self.tanh(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output


# 一个encoder从输入到输出全过程
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()
        self.gru = nn.GRU(input_size=d_model,
                            hidden_size=d_model,
                            num_layers=3,
                            dropout=p_drop,
                            bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2,d_model * 2)
        self.batchnorm1d = nn.BatchNorm1d(d_model * 2)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model * 2,d_model)




        self.layernorm1 = nn.LayerNorm(d_model*2, eps=1e-5)

        # self.ffn = PositionWiseFeedForwardNetwork(d_model*2, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model*2, eps=1e-5)


    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        attn_outputs, ht = self.gru(inputs)
        attn_outputs = attn_outputs.view(-1,attn_outputs.size(2))
        attn_outputs = self.linear1(attn_outputs)
        attn_outputs = self.batchnorm1d(attn_outputs)
        attn_outputs = self.tanh(attn_outputs)
        attn_outputs = self.dropout(attn_outputs)
        attn_outputs = self.linear2(attn_outputs)
        attn_outputs = attn_outputs.view(inputs.size(0),inputs.size(1),-1)
        attn_outputs = attn_outputs+inputs
        return attn_outputs


# 整个encoder的过程，包含多个encoder
class bigru_encoder(nn.Module):
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

    def __init__(self, config, d_model=300, n_layers=1, n_heads=6, d_ff=2048, pad_id=0):
        super(bigru_encoder, self).__init__()
        self.config = config
        p_drop = config.rnn_fc_dropout
        target_size = config.label
        self.pad_id = pad_id
        # layers
        self.embedding = nn.Embedding(config.words_num, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        # self.layers = EncoderLayer(d_model, n_heads, p_drop, d_ff)
        # layers to classify

        self.linear = nn.Linear(d_model, target_size)




    def forward(self, inputs):

        # self.config = config
        inputs = torch.arange(0, inputs.text.size(0), step=1, device='cpu').repeat(inputs.text.size(1), 1) + 1
        # |inputs| : (batch_size, seq_len)
        outputs = self.embedding(inputs)
        # |outputs| : (batch_size, seq_len, d_model)
        # attention_weights = []
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = self.linear(outputs)

        scores = torch.mean(outputs, dim=1)
        return scores

