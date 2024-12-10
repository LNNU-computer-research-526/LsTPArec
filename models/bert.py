from .base import BaseModel
from .bert_modules.bert import BERT
from .bert_modules.utils.layer_norm import LayerNorm
import torch
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.bert_hidden_units = args.bert_hidden_units
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert_hidden_units, args.num_items + 1)
        self.max_len = args.bert_max_len
        self.norm = LayerNorm(self.max_len)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        c_i= self.bert(x)[-1]
        rec_output = self.out(c_i)
        return rec_output, c_i


