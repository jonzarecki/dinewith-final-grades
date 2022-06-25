import torch
import torch.nn as nn
from common.constants import MODEL_LONG_TERM_DIR
from transformers import BertModel


class BertGRU(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=256, n_layers=2, bidirectional=True, dropout=0.25):

        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", cache_dir=f"{MODEL_LONG_TERM_DIR}/bert-base-uncased/"
        )
        # don't train bert params
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        embedding_dim = self.bert.config.to_dict()["hidden_size"]

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output
