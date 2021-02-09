import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy

from modules.relative_transformer import RelativeMultiHeadAttn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, Q, K, V):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(1)
        residual = Q
        # print('Q是不是tensor:')
        # print(torch.is_tensor(Q))

        q_heads = self.WQ(Q).view(batch_size, Q.size(0), self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, K.size(0), self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, V.size(0), self.n_heads, self.d_v).transpose(1, 2)
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        attn = self.scaled_dot_product_attn(q_heads, k_heads, v_heads)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(-1, batch_size, self.n_heads * self.d_v)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        output = self.dropout(output)
        output += residual
        # |output| : (batch_size, q_len, d_model)
        # print('output是不是tensor:')
        # print(torch.is_tensor(output))

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """
        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask):
        """
        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """


        # print('Multihead:')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        batch_size = x.size(1)
        max_len = x.size(0)
        d_model = x.size(2)

        x = nn.Linear(d_model,3*d_model,bias=False)(x)



        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale
        # print('Multihead:')
        # print(attn.size(0))
        # print(attn.size(1))
        # print(attn.size(2))
        # print(attn.size(3))
        #
        # attn.masked_fill_(mask=mask[:, None].eq(0), value=float('-inf'))
        # print('Multihead:')
        # print(attn.size(0))
        # print(attn.size(1))
        # print(attn.size(2))
        # print(attn.size(3))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        # print('Multihead:')
        # print(attn.size(0))
        # print(attn.size(1))
        # print(attn.size(2))
        # print(attn.size(3))
        # print('Multihead:')
        # print(v.size(0))
        # print(v.size(1))
        # print(v.size(2))
        # print(v.size(3))
        attn = self.dropout_layer(attn)

        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)
        # print('Multihead:')
        # print(v.size(0))
        # print(v.size(1))
        # print(v.size(2))



        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        """
        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)



        x = self.self_attn(x, mask)
        x = x.transpose(0,1)
        # print('TransformerLAyer')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        x = x + residual
        # print('TransformerLAyer')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        # print('??????????????')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))

        for layer in self.layers:
            x = layer(x, mask)

        # print('^^^^^^^^^^^^^^^^^^^')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        return x


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        # print('/////////////////')
        # print(emb.size(0))
        # print(emb.size(1))

        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz = input.size(0)
        seq_len = input.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)

        pos = self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

        pos = pos.contiguous().view(-1,pos.size(2)).detach()
        pos = nn.Linear(90000,300)(pos)
        pos = pos.view(bsz,seq_len,-1)
        # print('/////////////////')
        # print(pos.size(0))
        # print(pos.size(1))
        # print(pos.size(2))


        return pos

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)


class Tentest2(nn.Module):
    def __init__(self, config):
        super(Tentest2, self).__init__()
        self.config = config
        target_size = config.label
        self.embed = nn.Embedding(config.words_num, config.words_dim)

        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        self.pad_id = 0

        self.gru = nn.GRU(input_size=config.words_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          dropout=config.rnn_dropout,
                          bidirectional=True)

        self.transformerEncoder = TransformerEncoder(num_layers=1, d_model=300, n_head=1, feedforward_dim=300,
                                                     dropout=0.3, after_norm=True,attn_type='transformer',scale=False,dropout_attn=None,pos_embed='sin')

        Ks = 3
        self.mha = MultiHeadAttention(config.hidden_size * 2, 1)
        self.cnnmha = MultiHeadAttention(config.hidden_size, 1)

        self.linear1 = nn.Linear(config.hidden_size * 2, 2048)
        self.batchnorm1d = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(2048, target_size)

        self.linear3 = nn.Linear(Ks * config.output_channel, 2048)
        self.batchnormld2 = nn.BatchNorm1d(2048)

        self.linear4 = nn.Linear(600, 250)

        self.fc = nn.Linear(300,600)

        input_channel = 1
        self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
        self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
        self.dropoutcnn = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, target_size)
        self.cnnbatchnorm = nn.BatchNorm1d(250)

    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        Transformer_input = text
        x = self.embed(text)
        num_word, batch_size, words_dim = x.size()

        mask = x.ne(0)

        # Bilstm+Multihead_Attention

        outputs, ht = self.gru(x)
        outputs = self.mha(outputs, outputs, outputs)

        # outputs = outputs.view(-1, outputs.size(2))
        # outputs = self.linear1(outputs)
        # outputs = self.batchnorm1d(outputs)
        # outputs = self.relu(outputs)
        # outputs = self.dropout(outputs)
        # outputs = self.linear2(outputs)
        # tags = outputs.view(num_word, batch_size, -1)
        # scores1 = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)

        # CNN

        # x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        # conv1 = F.relu(self.conv1(x)).squeeze(3)
        # conv1 = conv1.transpose(1, 2).contiguous()
        # conv1 = conv1.transpose(0, 1).contiguous()
        # conv2 = F.relu(self.conv2(x)).squeeze(3)
        # conv2 = conv2.transpose(1, 2)
        # conv2 = conv2.transpose(0, 1)
        # conv3 = F.relu(self.conv3(x)).squeeze(3)
        # conv3 = conv3.transpose(1, 2)
        # conv3 = conv3.transpose(0, 1)
        # conv1 = self.cnnmha(conv1, conv1, conv1)
        # conv1 = conv1.transpose(0, 1).contiguous()
        # conv1 = conv1.transpose(1, 2).contiguous()
        # conv2 = self.cnnmha(conv2, conv2, conv2)
        # conv2 = conv2.transpose(0, 1).contiguous()
        # conv2 = conv2.transpose(1, 2).contiguous()
        # conv3 = self.cnnmha(conv3, conv3, conv3)
        # conv3 = conv3.transpose(0, 1).contiguous()
        # conv3 = conv3.transpose(1, 2).contiguous()
        # x = [conv1, conv2, conv3]
        # # (batch, channel_output, ~=sent_len) * Ks
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # # (batch, channel_output) * Ks
        # x = torch.cat(x, 1)  # (batch, channel_output * Ks)
        # x = self.dropoutcnn(x)
        # x = self.linear3(x)
        # x = self.batchnormld2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # logit = self.fc1(x)  # (batch, target_size)
        # scores2 = logit


        # TransformerEncoder
        # Transformer_input = Transformer_input.transpose(0,1).contiguous()
        Transformer_input = self.embed(Transformer_input)
        # Transformer_input = self.fc(Transformer_input)
        # print('@@@@@@@@@')
        # print(Transformer_input.size(0))
        # print(Transformer_input.size(1))
        # print(Transformer_input.size(2))

        outputs = nn.Linear(600,300)(outputs)


        transformer = self.transformerEncoder(outputs, mask)
        scores3 = nn.functional.normalize(torch.mean(transformer, dim=0), dim=1)
        # print('1818181818181818181')
        # print(scores3.size(0))
        # print(scores3.size(1))
        #
        # print('........')
        # print(scores1.size(0))
        # print(scores1.size(1))


        # scores = scores1 + scores2 + scores3

        scores = nn.functional.normalize(scores3, dim=1)
        scores = nn.Linear(300,250)(scores)
        return scores