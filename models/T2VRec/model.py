import numpy as np
import torch
import sys
from Time2Vec.periodic_activations import SineActivation, CosineActivation
import torch.nn as nn
import torch.nn.functional as F

FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):  # wried, why fusion X 2?
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Time2vec(nn.Module):
    def __init__(self, c_in, c_out, activation="cos"):
        super().__init__()
        self.wnbn = nn.Linear(c_in, c_out - 1, bias=True)
        self.w0b0 = nn.Linear(c_in, 1, bias=True)
        self.act = torch.cos if activation == "cos" else torch.sin

    def forward(self, x):
        part0 = self.act(self.w0b0(x))
        part1 = self.act(self.wnbn(x))
        return torch.cat([part0, part1], -1)


class TimeEncoder_abs(torch.nn.Module):
    def __init__(self, args):
        super(TimeEncoder_abs, self).__init__()
        # Set the periods for absolute and relative time
        self.P_abs = 12
        # The vector phi is a learned parameter
        self.phi = nn.Parameter(torch.Tensor(2 * self.P_abs))
        # Manually chosen periods of real-life importance
        self.abs_periods = (torch.tensor([0.25, 0.5, 0.75, 1, 2, 4, 8, 16, 24, 168, 672, 8760],
                                         dtype=torch.float32) * 3600).to(args.device)
        # Log scale periods for relative time difference

    def forward(self, t):
        # Calculate the features for absolute and relative time
        r_abs = self.encode_time(t, self.P_abs, self.abs_periods)
        return r_abs

    def encode_time(self, t, P, periods):
        t = t.float()
        periods = periods.unsqueeze(0).unsqueeze(0)
        features = []
        for i in range(1, P + 1):
            pi = periods[..., i - 1]
            phi_2i_minus_1 = self.phi[2 * i - 2]
            phi_2i = self.phi[2 * i - 1]
            features.append(torch.cos(2 * np.pi * t / pi + phi_2i_minus_1))
            features.append(torch.sin(2 * np.pi * t / pi + phi_2i))
        features.append(torch.log1p(t))
        return torch.cat(features, dim=-1)


class TimeEncoder_rel(TimeEncoder_abs):
    def __init__(self, args):
        super(TimeEncoder_rel, self).__init__(args)
        # Set the periods for absolute and relative time
        self.P_rel = 32
        # The vector phi is a learned parameter
        self.phi = nn.Parameter(torch.Tensor(2 * self.P_rel))
        # Manually chosen periods of real-life importance
        self.rel_periods = torch.logspace(np.log10(1), np.log10(4*7*24*3600), self.P_rel).float()
        # Log scale periods for relative time difference

    def forward(self, t):
        # Calculate the features for absolute and relative time
        r_abs = self.encode_time(t, self.P_rel, self.rel_periods)
        return r_abs



# class Time2Vec_abs(torch.nn.Module):
#     def __init__(self, args):
#         super(Time2Vec_abs, self).__init__()
#         self.dev = args.device
#         self.periods = torch.tensor(
#             [0.25 * 3600, 0.5 * 3600, 0.75 * 3600, 1 * 3600, 2 * 3600, 4 * 3600, 8 * 3600, 16 * 3600, 24 * 3600
#                 , 7 * 24 * 3600, 28 * 24 * 3600, 365 * 24 * 3600], dtype=torch.float32, requires_grad=False)
#         self.P = len(self.periods)
#         self.phase_sin = torch.nn.Parameter(torch.randn(self.P), requires_grad=True)
#         self.phase_cos = torch.nn.Parameter(torch.randn(self.P), requires_grad=True)
#
#     def forward(self, time_seq):
#         # absolute time features
#         # time_seq.shape = [batch_size, num_seq, 1]
#         batch_size = time_seq.shape[0]
#         seq_len = time_seq.shape[1]
#         periods_exp = self.periods.unsqueeze(0).unsqueeze(0)
#         # periods_exp.shape = [1, 1, P]
#         periods_exp = periods_exp.to(self.dev)
#         time_scaled = time_seq / periods_exp
#         # [batch_size, num_seq, P]
#         features_cos = torch.cos((2 * np.pi * time_scaled) + self.phase_cos)
#         # [batch_size, num_seq, P]
#         features_sin = torch.sin((2 * np.pi * time_scaled) + self.phase_sin)
#         # [batch_size, num_seq, P]
#         features_log = torch.log(time_seq)
#         # [batch_size, num_seq, 1]
#         stacked_features = torch.stack((features_cos, features_sin), dim=-1)
#         stacked_features = stacked_features.view(batch_size, seq_len, -1)
#         features_log = features_log
#         concat_features = torch.cat([stacked_features, features_log], dim=-1)
#         return concat_features


# class Time2Vec_rel(torch.nn.Module):
#     def __init__(self, periods):
#         super(Time2Vec_rel, self).__init__()
#         self.P = periods
#         self.periods = torch.logspace(np.log10(1), np.log10(4 * 7 * 24 * 3600), self.P)
#         self.phase_sin = torch.nn.Parameter(torch.randn(self.P), requires_grad=True)
#         self.phase_cos = torch.nn.Parameter(torch.randn(self.P), requires_grad=True)
#
#     def forward(self, time_seq):
#         # relative time features
#         batch_size = time_seq.shape[0]
#         seq_len = time_seq.shape[1]
#         time_seq = time_seq.unsqueeze(-1)
#         periods_exp = self.periods.unsqueeze(0).unsqueeze(0)
#         time_scaled = time_seq / periods_exp  # divide by the periods
#         features_cos = torch.cos((2 * np.pi * time_scaled) + self.phase_cos)
#         features_sin = torch.sin((2 * np.pi * time_scaled) + self.phase_sin)
#         features_log = torch.log(time_seq)
#         stacked_features = torch.stack((features_cos, features_sin), dim=-1)
#         stacked_features = stacked_features.view(batch_size, seq_len, -1)
#         features_log = features_log
#         concat_features = torch.cat([stacked_features, features_log], dim=-1)
#         return concat_features

class T2V_SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(T2V_SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # self.t2v_cos = Time2vec(1, args.hidden_units, activation='cos')
        # self.t2v_sin = Time2vec(1, args.hidden_units, activation='sin')
        self.t2v_abs = TimeEncoder_abs(args)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # Linear layer to project the combined vector
        # self.projection = torch.nn.Linear(args.hidden_units * 2, args.hidden_units)
        self.projection = torch.nn.Linear(args.hidden_units + 25, args.hidden_units)
        # Two-layer MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, 4 * args.hidden_units),
            torch.nn.GELU(),
            torch.nn.Linear(4 * args.hidden_units, args.hidden_units),
        )
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq2feats(self, log_seqs, time_seqs):
        # get the item embedding
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, seq_len, embeddings]
        # time_seqs.shape = [batch_size, seq_len]
        # t_seqs = self.t2v_abs(torch.LongTensor(time_seqs).to(self.dev))
        # print(t_seqs.shape)
        # print(t_seqs)
        # print(time_seqs[-1, -1])
        t_seqs = self.t2v_abs(torch.LongTensor(time_seqs).unsqueeze(-1).to(self.dev))
        # print(t_seqs[-1, -1])
        # t_seqs = self.t2v_cos(torch.FloatTensor(time_seqs).unsqueeze(-1).to(self.dev))
        # print(t_seqs)
        # t_seqs = self.t2v_sin(torch.FloatTensor(time_seqs).unsqueeze(-1).to(self.dev))
        # time features
        # t_seqs.shape = [batch_size, seq_len, num_features]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = torch.cat([seqs, t_seqs], -1)
        seqs = self.projection(seqs)
        # transform it back to hidden units
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
        # drop out before the time features or not?
        # dropout
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        # mask the elements, ignore the padding tokens (items with id 0) in the sequences.
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # casual masking
        # input the sequence to attention layers
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False this arg do not work?
            # multi-head attention outputs
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        log_feats = self.mlp(log_feats)
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    ## SASRec forward
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.seq2feats(log_seqs, time_seqs)  # user_ids hasn't been used yet
        # (batch_size, sequence_length, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)

        # score for negative engagement = (batch_size, sequence_length)
        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_seqs, item_indices):  # for inference
        log_feats = self.seq2feats(log_seqs, time_seqs)  # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        return logits  # preds # (U, I)


class T2V_AllAction(T2V_SASRec):  # similar to torch.nn.MultiheadAttention
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.seq2feats(log_seqs, time_seqs)
        final_feat = log_feats[:, -1, :]
        # features obtained: embedding including time features
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))[:, -1, :]
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))[:, -1, :]
        # print(final_feat.shape)
        # print(pos_embs.shape)
        pos_logits = (final_feat.unsqueeze(1) * pos_embs).sum(dim=-1)
        neg_logits = (final_feat.unsqueeze(1) * neg_embs).sum(dim=-1)
        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        return pos_logits, neg_logits  # pos_pred, neg_pred


class T2V_DenseAllAction(T2V_SASRec):  # similar to torch.nn.MultiheadAttention
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        # 1 pos, num_neg
        log_feats = self.seq2feats(log_seqs, time_seqs)
        # log_feats.shape = [batch_size, seq_len, hidden_units]
        # pos_seqs.shape = [batch_size, seq_len]
        # neg_seqs.shape = [batch_size, seq_len]
        # get the last item in the sequence
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs.shape = (batch_size, seq_len, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_embs.shape = (batch_size, seq_len, neg_nums, hidden_units)
        # print(neg_embs.shape)
        # print(log_feats.shape)
        # print(log_feats.unsqueeze(2).shape)
        pos_embs = pos_embs.squeeze(-2)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # pos_logits.shape = (batch_size, seq_len)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        # log_feats.unsqueeze(2).shape = [batch_size, seq_len, 1, hidden_units]
        # neg_logits.shape = (batch_size, seq_len, neg_nums)
        # pos_logits = pos_logits.unsqueeze(-1)
        return pos_logits, neg_logits


class T2V_DenseAllPlus(T2V_SASRec):  # similar to torch.nn.MultiheadAttention
    # Dense all +, dense all ++, combined
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        # num_pos, num_neg
        log_feats = self.seq2feats(log_seqs, time_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats.unsqueeze(2) * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits


class T2V_SASRecSampledLoss(T2V_SASRec):
    def __init__(self, user_num, item_num, args):
        super().__init__(user_num, item_num, args)
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))

    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.seq2feats(log_seqs, time_seqs)  # user_ids hasn't been used yet
        # (batch_size, sequence_length, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        pos_logits = (log_feats * pos_embs).sum(dim=-1) / self.temperature
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # score for negative engagement = (batch_size, sequence_length)
        return pos_logits, neg_logits  # pos_pred, neg_pred


class T2V_AllActionSampledLoss(T2V_SASRecSampledLoss):
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.seq2feats(log_seqs, time_seqs)
        final_feat = log_feats[:, -1, :]
        # features obtained: embedding including time features
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))[:, -1, :]
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))[:, -1, :]
        # print(final_feat.shape)
        # print(pos_embs.shape)
        pos_logits = (final_feat.unsqueeze(1) * pos_embs).sum(dim=-1) / self.temperature
        neg_logits = (final_feat.unsqueeze(1) * neg_embs).sum(dim=-1) / self.temperature
        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        return pos_logits, neg_logits  # pos_pred, neg_pred


class T2V_DenseAllActionSampledLoss(T2V_SASRecSampledLoss):
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        # 1 pos, num_neg
        log_feats = self.seq2feats(log_seqs, time_seqs)
        # log_feats.shape = [batch_size, seq_len, hidden_units]
        # pos_seqs.shape = [batch_size, seq_len]
        # neg_seqs.shape = [batch_size, seq_len]
        # get the last item in the sequence
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs.shape = (batch_size, seq_len, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_embs.shape = (batch_size, seq_len, neg_nums, hidden_units)
        # print(neg_embs.shape)
        # print(log_feats.shape)
        # print(log_feats.unsqueeze(2).shape)
        pos_embs = pos_embs.squeeze(-2)
        pos_logits = (log_feats * pos_embs).sum(dim=-1) / self.temperature
        # pos_logits.shape = (batch_size, seq_len)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # log_feats.unsqueeze(2).shape = [batch_size, seq_len, 1, hidden_units]
        # neg_logits.shape = (batch_size, seq_len, neg_nums)
        # pos_logits = pos_logits.unsqueeze(-1)
        return pos_logits, neg_logits


class T2V_DenseAllPlusSampledLoss(T2V_SASRecSampledLoss):
    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        # num_pos, num_neg
        log_feats = self.seq2feats(log_seqs, time_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats.unsqueeze(2) * pos_embs).sum(dim=-1) / self.temperature
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        return pos_logits, neg_logits
