import torch
from torch import nn
import torch.nn.functional as F


class bi_bigru_ffn(nn.Module):
    def __init__(self, config):
        super(bi_bigru_ffn, self).__init__()
        self.config = config
        target_size = config.label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False

        self.gru = nn.GRU(input_size=config.words_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          dropout=config.rnn_dropout,
                          bidirectional=True)
        self.linear1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.batchnorm1d = nn.BatchNorm1d(config.hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.linear2 = nn.Linear(config.hidden_size * 2, target_size)




    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        x = self.embed(text)
        num_word, batch_size, words_dim = x.size()
        # h0 / c0 = (layer*direction, batch_size, hidden_dim)

        outputs, ht = self.gru(x)
        outputs = outputs.view(-1, outputs.size(2))
        outputs = self.linear1(outputs)
        outputs = self.batchnorm1d(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)

        tags = outputs.view(num_word, batch_size, -1)
        scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)


        return scores


