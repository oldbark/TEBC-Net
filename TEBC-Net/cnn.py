import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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

class cnn(nn.Module):
    def __init__(self, config):
        super(cnn, self).__init__()
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


        Ks = 3
        self.mha = MultiHeadAttention(config.hidden_size * 2, 1)
        self.cnnmha = MultiHeadAttention(config.hidden_size, 1)

        self.linear1 = nn.Linear(config.hidden_size * 2, 2048)
        self.batchnorm1d = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(2048, target_size)

        self.linear3 = nn.Linear(Ks * config.output_channel,2048)
        self.batchnormld2 = nn.BatchNorm1d(2048)

        self.linear4 = nn.Linear(600,250)


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
        x = self.embed(text)
        num_word, batch_size, words_dim = x.size()

        #Bilstm+Multihead_Attention
        outputs, ht = self.gru(x)
        outputs = self.mha(outputs, outputs, outputs)

        outputs = outputs.view(-1, outputs.size(2))
        outputs = self.linear1(outputs)
        outputs = self.batchnorm1d(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        tags = outputs.view(num_word, batch_size, -1)
        scores1 = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)



        #CNN
        # print('x:')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        # print('x:')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        # print(x.size(3))
        # print('sss')
        # print(F.relu(self.conv1(x)).size(0))
        # print(F.relu(self.conv1(x)).size(1))
        # print(F.relu(self.conv1(x)).size(2))
        # print(F.relu(self.conv1(x)).size(3))
        conv1 = F.relu(self.conv1(x)).squeeze(3)
        # print('conv1:')
        # print(conv1.size(0))
        # print(conv1.size(1))
        # print(conv1.size(2))
        
        conv1 = conv1.transpose(1,2).contiguous()
        conv1 = conv1.transpose(0,1).contiguous()

        conv2 = F.relu(self.conv2(x)).squeeze(3)
        conv2 = conv2.transpose(1, 2)
        conv2 = conv2.transpose(0,1)

        conv3 = F.relu(self.conv3(x)).squeeze(3)
        conv3 = conv3.transpose(1, 2)
        conv3 = conv3.transpose(0,1)


        conv1 = self.cnnmha(conv1,conv1,conv1)
        conv1 = conv1.transpose(0, 1).contiguous()
        conv1 = conv1.transpose(1, 2).contiguous()
        conv2 = self.cnnmha(conv2,conv2,conv2)
        conv2 = conv2.transpose(0, 1).contiguous()
        conv2 = conv2.transpose(1, 2).contiguous()
        conv3 = self.cnnmha(conv3,conv3,conv3)
        conv3 = conv3.transpose(0, 1).contiguous()
        conv3 = conv3.transpose(1, 2).contiguous()




        x = [conv1, conv2, conv3]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # (batch, channel_output) * Ks

        x = torch.cat(x, 1)  # (batch, channel_output * Ks)
        x = self.dropoutcnn(x)

        x = self.linear3(x)
        x = self.batchnormld2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        scores2 = logit
        #
        #
        # scores = scores1 + scores2
        # scores = nn.Linear(250,1024)(scores)
        # scores = nn.BatchNorm1d(1024)(scores)
        # scores = self.relu(scores)
        # scores = nn.Dropout(0.3)(scores)
        # scores = nn.Linear(1024,250)(scores)
        scores = scores1 + scores2

        scores = nn.functional.normalize(scores,dim=1)
        # scores = F.log_softmax(scores, dim=1)
        return scores






