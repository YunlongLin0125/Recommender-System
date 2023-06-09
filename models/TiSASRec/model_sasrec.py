import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
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


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    ## SASRec forward
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
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

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class SASRecSampledLoss(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRecSampledLoss, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    ## SASRec forward
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        # (batch_size, sequence_length, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, sequence_length, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # (batch_size, sequence_length, num_negs, hidden_units)
        pos_logits = (log_feats * pos_embs).sum(dim=-1) / self.temperature
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # score for negative engagement = (batch_size, sequence_length, num_negs)
        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class AllAction(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(AllAction, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        # final_feature.shape = [batch_size, hidden_units]
        # pos_seqs.shape = [batch_size, seq_len, num_target]
        # neg_seqs.shape = [batch_size, seq_len, num_neg]
        # get the last item in the sequence
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))[:, -1, :]
        # pos_embs.shape = (batch_size, num_target, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))[:, -1, :]
        # neg_embs.shape = (batch_size, num_negs, hidden_units)
        # len(pos_seqs)
        pos_logits = (final_feat.unsqueeze(1) * pos_embs).sum(dim=-1)
        # pos_logits.shape = (batch_size, num_target)
        neg_logits = (final_feat.unsqueeze(1) * neg_embs).sum(dim=-1)
        # final_feat.unsqueeze(1).shape = [batch_size, 1, hidden units]
        # neg_logits.shape = (batch_size, num_negs)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class AllActionSampledLoss(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(AllActionSampledLoss, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        # final_feature.shape = [batch_size, hidden_units]
        # pos_seqs.shape = [batch_size, seq_len, num_target]
        # neg_seqs.shape = [batch_size, seq_len, num_neg]
        # get the last item in the sequence
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))[:, -1, :]
        # pos_embs.shape = (batch_size, num_target, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))[:, -1, :]
        # neg_embs.shape = (batch_size, num_negs, hidden_units)
        # len(pos_seqs)
        pos_logits = (final_feat.unsqueeze(1) * pos_embs).sum(dim=-1) / self.temperature
        # pos_logits.shape = (batch_size, num_target)
        neg_logits = (final_feat.unsqueeze(1) * neg_embs).sum(dim=-1) / self.temperature
        # final_feat.unsqueeze(1).shape = [batch_size, 1, hidden units]
        # neg_logits.shape = (batch_size, num_negs)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class DenseAll(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DenseAll, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
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
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class DenseAllSampledLoss(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DenseAllSampledLoss, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        # log_feats.shape = [batch_size, seq_len, hidden_units]
        # pos_seqs.shape = [batch_size, seq_len]
        # neg_seqs.shape = [batch_size, seq_len]
        # get the last item in the sequence
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs.shape = (batch_size, seq_len, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_embs.shape = (batch_size, seq_len, neg_nums, hidden_units)
        # print(log_feats.unsqueeze(2).shape)
        pos_embs = pos_embs.squeeze(-2)
        pos_logits = (log_feats * pos_embs).sum(dim=-1) / self.temperature
        # score for positive engagement = (batch_size, sequence_length)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # score for negative engagement = (batch_size, sequence_length, num_negs)
        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class DenseAllPlus(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DenseAllPlus, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        # log_feats.shape = [batch_size, seq_len, hidden_units]
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs.shape = (batch_size, seq_len, num_pos, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_embs.shape = (batch_size, seq_len, num_neg, hidden_units)
        pos_logits = (log_feats.unsqueeze(2) * pos_embs).sum(dim=-1)
        # pos_logits.shape = (batch_size, seq_len, num_pos)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1)
        # neg_logits.shape = (batch_size, seq_len, num_neg)
        # log_feats.unsqueeze(2).shape = [batch_size, seq_len, 1]

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)


class DenseAllPlusSampledLoss(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DenseAllPlusSampledLoss, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        initial_temperature = 1.0  # you can choose any initial value depending on your problem
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature], device=self.dev))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # get the item embedding
        # seqs.shape = [batch_size, 1]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs.shape = [batch_size, embeddings]
        # scale the values
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # position embeddings
        seqs = self.emb_dropout(seqs)
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
        # (batch_size, sequence_length, hidden_units)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        # log_feats.shape = [batch_size, seq_len, hidden_units]
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs.shape = (batch_size, seq_len, num_pos, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_embs.shape = (batch_size, seq_len, num_neg, hidden_units)
        pos_logits = (log_feats.unsqueeze(2) * pos_embs).sum(dim=-1) / self.temperature
        # pos_logits.shape = (batch_size, seq_len, num_pos)
        neg_logits = (log_feats.unsqueeze(2) * neg_embs).sum(dim=-1) / self.temperature
        # neg_logits.shape = (batch_size, seq_len, num_neg)
        # log_feats.unsqueeze(2).shape = [batch_size, seq_len, 1]

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # print(item_embs.shape)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
