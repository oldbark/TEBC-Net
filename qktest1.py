import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

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

class qktest1(nn.Module):
    def __init__(self, config):
        super(qktest1, self).__init__()
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

        self.mha1 = MultiHeadAttention(config.hidden_size * 2, 1)
        self.mha2 = MultiHeadAttention(config.hidden_size * 2, 1)
        self.linear1 = nn.Linear(config.hidden_size * 2, 2048)
        self.batchnorm1d = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.linear2 = nn.Linear(2048, target_size)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, config.channel_size, (config.conv_kernel_1, config.conv_kernel_2), stride=1,
        #               padding=(config.conv_kernel_1 // 2, config.conv_kernel_2 // 2)),
        #     # channel_in=1, channel_out=8, kernel_size=3*3
        #     nn.ReLU(True))
        # self.seq_maxlen = config.seq_maxlen + (config.conv_kernel_1 + 1) % 2
        # self.rel_maxlen = config.rel_maxlen + (config.conv_kernel_2 + 1) % 2
        #
        # self.pooling = nn.MaxPool2d(kernel_size=(1, 1),
        #                             stride=(1, 1), padding=0)
        # self.fc = nn.Sequential(
        #     nn.Linear(config.rel_maxlen * config.channel_size, 20),
        #     nn.ReLU(),
        #     nn.Dropout(p=config.rnn_fc_dropout),
        #     nn.Linear(20, 1))




    def forward(self, x,rel):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        x = self.embed(text)
        num_word, batch_size, words_dim = x.size()

        # rel = self.embed(rel).unsqueeze(1)
        #(32,1,300)


        outputs, ht = self.gru(x)

        outputs = self.mha1(outputs, outputs, outputs)
        # outputs = self.mha2(outputs,outputs,outputs)
        outputs = outputs.view(-1, outputs.size(2))
        outputs = self.linear1(outputs)
        outputs = self.batchnorm1d(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)



        # inputs = x.view(x.size(1), x.size(0), -1)
        # out = self.matchPyramid(inputs, rel, inputs.size(1), rel.size(1))
        tags = outputs.view(num_word, batch_size, -1)
        scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)



        # rel = rel.unsqueeze(0)
        # print('rel:')
        # print(rel.size(0))
        # print(rel.size(1))
        # print(rel.size(2))
        # relation = self.gru1(rel)
        # print('relation: ')
        # print(relation.size(0))
        # print(relation.size(1))
        # print(relation.size(2))
        # relation = relation.view(-1, relation.size(2))
        # relation = self.linear1(relation)
        # relation = self.batchnorm1d(relation)
        # relation = self.relu(relation)
        # relation = self.dropout(relation)
        # relation = self.linear2(relation)
        # tags2 = relation.view(rel.size(0),batch_size,-1)
        # scores2 = nn.functional.normalize(torch.mean(tags2,dim=0),dim=1)

        # outoutput = torch.cat((scores, out), 1)
        # outoutput = nn.Linear(250,250)(outoutput)


        return scores


    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(0), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def matchPyramid(self, seq, rel, seq_len, rel_len):
        '''
        param:
            seq: (batch, _seq_len, embed_size)
            rel: (batch, _rel_len, embed_size)
            seq_len: (batch,)
            rel_len: (batch,)
        return:
            score: (batch, 1)
        '''
        batch_size = seq.size(0)

        rel_trans = torch.transpose(rel, 1, 2)
        # (batch, 1, seq_len, rel_len)
        seq_norm = torch.sqrt(torch.sum(seq * seq, dim=2, keepdim=True))
        rel_norm = torch.sqrt(torch.sum(rel_trans * rel_trans, dim=1, keepdim=True))
        cross = torch.bmm(seq / seq_norm, rel_trans / rel_norm).unsqueeze(1)


        # (batch, channel_size, seq_len, rel_len)

        conv1 = self.conv(cross)
        channel_size = conv1.size(1)



        # (batch, channel_size, p_size1, p_size2)
        pool1 = self.pooling(conv1).view(batch_size,-1)
        pool1 = nn.Linear(conv1.size(2) * channel_size, 20)(pool1)
        pool1 = self.relu(pool1)
        pool1 = self.dropout(pool1)
        pool1 = nn.Linear(20, 20)(pool1)

        # (batch, 1)
        # out = self.fc(pool1)


        return pool1

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):

            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            for i in range(max_len1):
                idx1_one = int(i/stride1)
            for i in range(max_len2):
                idx2_one = int(i/stride2)
            return idx1_one, idx2_one
        batch_size = len1
        index1, index2 = [], []
        for i in range(batch_size):
            idx1_one, idx2_one = dpool_index_(i, len1, len2, max_len1, max_len2)
            index1.append(idx1_one)
            index2.append(idx2_one)
        index1 = torch.LongTensor(index1)
        index2 = torch.LongTensor(index2)
        if self.config.cuda:
            index1 = index1.cuda()
            index2 = index2.cuda()
        return Variable(index1), Variable(index2)


