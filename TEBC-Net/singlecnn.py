import torch
from torch import nn
import torch.nn.functional as F


class singlecnn(nn.Module):
    def __init__(self, config):
        super(singlecnn, self).__init__()
        self.config = config
        target_size = config.label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False

        if config.qa_mode.upper() == "CNN":
            input_channel = 1
            Ks = 3
            self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(Ks * config.output_channel, target_size)



    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        x = self.embed(text)


        x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1)  # (batch, channel_output * Ks)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        scores = nn.functional.normalize(logit, dim=1)

        return scores



