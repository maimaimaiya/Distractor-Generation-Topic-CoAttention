""" Hierarchical attention modules """
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd


class HierarchicalAttention(nn.Module):
    """Dynamic attention"""
    def __init__(self, dim, attn_type="general"):
        super(HierarchicalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        # Hierarchical attention
        if self.attn_type == "general":
            self.word_linear_in = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in = nn.Linear(dim, dim, bias=False)
        else:
            raise NotImplementedError

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s, type):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()
        if type == 'word':
            h_t_ = self.word_linear_in(h_t)
        elif type == 'sent':
            h_t_ = self.sent_linear_in(h_t)
        else:
            raise NotImplementedError
        h_t = h_t_.view(tgt_batch, 1, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, word_bank, word_lengths,
                sent_bank, sent_lengths,sent_context, static_attn):

        # source = source.unsqueeze(1)
        word_max_len, word_batch, words_max_len, word_dim = word_bank.size()
        sent_max_len, sent_batch, sent_dim = sent_bank.size()
        assert word_batch == sent_batch
        assert words_max_len == sent_max_len
        target_batch, target_dim = source.size()

        # reshape for compute word score
        # (word_max_len, word_batch, words_max_len, word_dim) -> transpose
        # (word_batch, word_max_len, words_max_len, word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (word_batch, words_max_len, word_max_len, word_dim)
        word_bank = word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            word_batch, words_max_len * word_max_len, word_dim)
        word_align = self.score(source, word_bank, 'word')

        # sentence score
        # (sent_batch, target_l, sent_max_len)
        sent_bank = sent_bank.transpose(0, 1).contiguous()
        sent_align = self.score(source, sent_bank, 'sent')

        # attn
        # align = (word_align.view(word_batch, 1, words_max_len, word_max_len) * sent_align.unsqueeze(-1) *\
        #               static_attn.unsqueeze(1).unsqueeze(-1)).view(word_batch, 1, words_max_len * word_max_len)
        align = (sent_align.unsqueeze(-1) * static_attn.unsqueeze(1).unsqueeze(-1)).view(word_batch, 1, words_max_len * word_max_len)

        mask = sequence_mask(word_lengths.view(-1), max_len=word_max_len).view(
            word_batch, words_max_len * word_max_len).unsqueeze(1)
        align.masked_fill_(~mask.cuda(), -float('inf'))#~mask.cuda()
        align_vectors = self.softmax(align) + 1e-20
        c = torch.bmm(align_vectors, word_bank).squeeze(1)
        concat_c = torch.cat([c, source], -1).view(target_batch, target_dim * 2)
        attn_h = self.linear_out(concat_c).view(target_batch, target_dim)
        attn_h = self.tanh(attn_h)
        return attn_h, align_vectors.squeeze(1)

class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(CQAttention,self).__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q):#, Cmask, Qmask):
        C = C.transpose(1, 0)
        Q = Q.transpose(1, 0)
        # Qmask.transpose(0, 1)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        # Cmask = Cmask.view(batch_size_c, Lc, 1)
        # Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(S, dim=2)#mask_logits(S, Qmask),
        S2 = F.softmax(S, dim=1)#mask_logits(S, Cmask),
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res
