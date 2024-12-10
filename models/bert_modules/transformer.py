import torch.nn as nn
import torch
from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward



class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, args, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()

        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        self.attention1 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        self.attention_r1 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        self.attention_r2 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        self.attention_r3 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        # self.attention_r4 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        #
        # self.attention_r5 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        #
        # self.attention_r6 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        self.feed_forward1 = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        self.input_sublayer_r1 = SublayerConnection(size=hidden, dropout=dropout)

        self.input_sublayer_r2 = SublayerConnection(size=hidden, dropout=dropout)

        self.input_sublayer_r3 = SublayerConnection(size=hidden, dropout=dropout)

        # self.input_sublayer_r4 = SublayerConnection(size=hidden, dropout=dropout)
        #
        # self.input_sublayer_r5 = SublayerConnection(size=hidden, dropout=dropout)
        #
        # self.input_sublayer_r6 = SublayerConnection(size=hidden, dropout=dropout)

        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.input = SublayerConnection(size=hidden, dropout=dropout)

        self.output = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask, r1, r2, r3, mask1, mask2, mask3):
        r1 = self.input_sublayer_r1(r1, lambda _r1: self.attention_r1.forward(_r1, _r1, _r1, mask=mask1))

        r2 = self.input_sublayer_r2(r2, lambda _r2: self.attention_r2.forward(_r2, _r2, _r2, mask=mask2))

        r3 = self.input_sublayer_r3(r3, lambda _r3: self.attention_r3.forward(_r3, _r3, _r3, mask=mask3))

        # r4 = self.input_sublayer_r4(r4, lambda _r4: self.attention_r4.forward(_r4, _r4, _r4, mask=mask4))
        #
        # r5 = self.input_sublayer_r5(r5, lambda _r5: self.attention_r5.forward(_r5, _r5, _r5, mask=mask5))
        #
        # r6 = self.input_sublayer_r6(r6, lambda _r6: self.attention_r6.forward(_r6, _r6, _r6, mask=mask6))

        r = torch.cat((r1, r2, r3), dim=1)

        r = self.input_sublayer(r, lambda _r: self.attention.forward(_r, _r, _r, mask=mask))

        r = self.output_sublayer(r, self.feed_forward)

        x = self.input(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))

        x = self.output(x, self.feed_forward1)

        out = 0.3 * r + 0.7 * x

        out = self.dropout(out)

        return out
