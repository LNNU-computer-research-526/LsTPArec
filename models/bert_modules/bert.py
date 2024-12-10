import torch
from torch import nn as nn
from .utils.layer_norm import LayerNorm
from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OtherLayer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class BERT(nn.Module):
    def __init__(self, args, fixed=False):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)

        self.max_len = args.bert_max_len
        self.dataset = args.dataset_code
        num_items = args.num_items

        n_layers = args.bert_num_blocks

        heads = args.bert_num_heads
        vocab_size = num_items + 2

        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        self.dataset = args.dataset_code

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=self.max_len,
                                       dropout=dropout)
        self.embedding1 = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=self.max_len,
                                        dropout=dropout)

        self.norm = LayerNorm(self.hidden)
        self.norm_sim = LayerNorm(self.max_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args, hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        ######fixed model
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, un_mask = x
        k_list = torch.split(x, 10, dim=1)
        masks = [(k > 0).unsqueeze(1).repeat(1, k.size(1), 1).unsqueeze(1) for k in k_list]
        mask1, mask2, mask3 = masks
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        un_mask = un_mask.to(device)
        x = self.embedding(x)
        un_mask = self.embedding1(un_mask)

        un_mask_t = un_mask.transpose(0, 2)
        un_mask_s = (un_mask.transpose(1, 2)).transpose(0, 1)
        x_s = (x.transpose(1, 2)).transpose(0, 1)
        similar = torch.matmul(un_mask_t, un_mask_s)

        similar = self.norm_sim(similar)
        similar = torch.sigmoid(similar)
        similar = torch.where(similar < 0.65, 0, similar)

        mat = torch.matmul(x_s, similar)
        mat = (mat.transpose(0, 1)).transpose(1, 2)
        mat = self.norm(mat)
        mat = mat + x
        r1, r2, r3 = torch.split(mat, 10, dim=1)

        layer_output = [x]

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, r1, r2, r3, mask1, mask2, mask3)
            layer_output.append(x)

        return layer_output

    def fix_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        pass
