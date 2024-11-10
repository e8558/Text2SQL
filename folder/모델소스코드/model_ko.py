import numpy as np
# import math

import torch
from torch import nn

import transformers
# from transformers import BertTokenizer, BertForQuestionAnswering
# from kobert import get_pytorch_kobert_model

import data_ko as Data
import utils_ko
# import utils


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def pool_vector(vector, pool_method="last"):
    "pooling dim from vector, 'first' method or 'average'"
    if pool_method == "first":
        return vector[0, :]
    elif pool_method == "last":
        return vector[-1, :]
    else:
        return torch.mean(vector, dim=0)


def extract_vectors(config, vector, q_len, c_len, t_len, pool_method="last"):
    "extract cls, query, cloumn vectors from encodes"
    batch_size = vector.size(0)

    h_v = vector[:, 0, :]  # cls token vectors
    max_q_len = max(q_len)  # max query length
    max_c_num = max([len(utils_ko.flat_list(c_)) for c_ in c_len])  # max total column number in schema
    max_c_len = max([len(c) for c in utils_ko.flat_list(c_len)])  # max column number in tables
    max_t_len = max(t_len)  # max table number in DB schema

    q_v = []  # query  vectors
    c_v = []  # column vectors
    t_v = []  # table  vectors
    q_mask = np.zeros((batch_size, max_q_len))  # question encoding mask
    c_mask = np.zeros((batch_size, max_c_num))  # column encoding mask
    t_mask = np.zeros((batch_size * max_t_len, max_c_len))  # table encoding mask
    d_mask = np.zeros((batch_size, max_t_len))  # schema encoding mask

    for i, (v, q, c) in enumerate(zip(vector, q_len, c_len)):

        # extract question token vectors
        l = 1  # skip cls token position
        p_v = v[l:l + q, :]
        q_mask[i, :q] = 1
        if q < max_q_len:
            p_v = torch.cat([p_v, torch.zeros(
                max_q_len - q
                , config.hidden_size
            ).to(v.device)])
        q_v.append(p_v.unsqueeze(0))

        l += (q + 1)  # add length (query length + one sep id)
        t_v_ = []
        c_t_v = []
        for t_i, t_c in enumerate(c):
            c_v_ = []
            for c_ in t_c:
                p_v = pool_vector(v[l:l + c_, :])
                c_v_.append(p_v.unsqueeze(0))
                l += (c_ + 1)
            c_t_v.extend(c_v_)
            c_v_ = torch.cat(c_v_)
            t_mask[i * max_t_len + t_i, :c_v_.size(0)] = 1
            if c_v_.size(0) < max_c_len:
                c_v_ = torch.cat([c_v_
                                     , torch.zeros(
                        (max_c_len - c_v_.size(0), config.hidden_size)
                    ).to(v.device)])
            t_v_.append(c_v_.unsqueeze(0))

        c_t_v = torch.cat(c_t_v)
        c_mask[i, :c_t_v.size(0)] = 1
        if c_t_v.size(0) < max_c_num:
            c_t_v = torch.cat([c_t_v
                                  , torch.zeros(
                    (max_c_num - c_t_v.size(0), config.hidden_size)
                ).to(v.device)])
        c_v.append(c_t_v.unsqueeze(0))

        t_v_ = torch.cat(t_v_)
        if t_v_.size(0) < max_t_len:
            t_v_ = torch.cat([t_v_
                                 , torch.zeros(
                    (max_t_len - t_v_.size(0), max_c_len, config.hidden_size)
                ).to(v.device)])

        t_v.append(t_v_.unsqueeze(0))
        d_mask[i, :len(c)] = 1

    device = vector.device
    q_v = torch.cat(q_v).to(device)
    c_v = torch.cat(c_v).to(device)
    t_v = torch.cat(t_v).to(device)
    q_mask = torch.from_numpy(q_mask).long().to(device)
    c_mask = torch.from_numpy(c_mask).long().to(device)
    t_mask = torch.from_numpy(t_mask).long().to(device)
    d_mask = torch.from_numpy(d_mask).long().to(device)

    return h_v, q_v, c_v, t_v, q_mask, c_mask, t_mask, d_mask


def tile(x, axis, repeat):
    "tile tensor with given axis / repeat numbers"
    repeats = [1 for _ in range(x.dim())] + [1]
    repeats[axis] = repeat
    repeats = tuple(repeats)
    return x.unsqueeze(axis).repeat(*repeats)


def transform(x, y):
    return torch.cat([x, y, torch.abs(x - y), x * y], dim=-1)


class Embedding(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        if encoder is None:
            # pretrain_config = transformers.AutoConfig.from_pretrained(
            #     "bert-large-uncased-whole-word-masking"
            #     , cache_dir="cache"
            # )
            # self.token_embedding = transformers.AutoModel.from_config(pretrain_config)

            # if config.pretrained_model == "monologg/kobert":
            #     print("\n\nload monologg kobert\n")
            #     self.token_embedding, _ = get_pytorch_kobert_model()
            # else:
            #     self.token_embedding = transformers.AutoModel.from_pretrained(
            #         config.pretrained_model
            #         , cache_dir="cache"
            #     )
            #     #
            #     # print(self.token_embedding)
            #     #
            self.token_embedding = transformers.AutoModel.from_pretrained(
                config.pretrained_model
                , cache_dir="cache"
            )

        def load(path, model):
            print("loading model from ", path)
            load_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(load_dict['model'])

    def forward(self, x, mask):
        out = self.token_embedding(x)
        if len(out) > 1:
            out = out[0]
        if type(out) == tuple:
            out = out[0]
        return out


class TableClause(nn.Module):
    "Network for table clause"

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)
        self.dense2 = nn.Linear(config.hidden_size, 1, bias=False)
        self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dense4 = nn.Linear(config.hidden_size, Data.MAX_NUM["table_num"])
        self.dense4 = nn.Linear(config.hidden_size, config.max_num["table_num"])
        self.dropout = nn.Dropout(config.dropout)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=Data.ignore_idx)
        self.config = config

    def forward(self, v_T, v_Q, v_D, v_P, d_mask, labels=None):
        batch_size = v_T.size(0)
        loss = 0
        preds = tuple()

        if labels is not None:
            table_num_label, table_id_label = labels
            table_id_label = table_id_label[:, :v_T.size(1)].contiguous()
            # print(table_num_label)
            # print(table_id_label)

        table_num = v_T.size(1)
        c = torch.cat([v_T
                          , tile(v_Q, 1, table_num)
                          , tile(v_D, 1, table_num)
                          , tile(v_P, 1, table_num)
                       ], dim=-1)

        table_id_logit = self.dense2(
            self.dropout(
                nn.Tanh()(
                    self.dense1(c)
                )
            )
        ).squeeze(-1)

        table_id_logit = table_id_logit.masked_fill(d_mask == 0, -1e9)
        # print(table_id_logit.view(-1))
        # print(table_id_label.view(-1).float())
        # print()
        # exit(-1)
        if labels is not None:
            loss = self.bce_loss(
                table_id_logit.view(-1)
                , table_id_label.view(-1).float()
            )
            loss = loss.view(-1) * d_mask.view(-1)

            loss = loss.view(batch_size, -1)
            loss = torch.sum(torch.sum(loss, dim=-1)) / batch_size

        preds += ((
                      to_cpu(nn.Sigmoid()(table_id_logit))
                      , None if labels is None else to_cpu(table_id_label)
                  ),)

        v_T_ = torch.matmul(nn.Softmax(dim=-1)(table_id_logit).unsqueeze(1), v_T).squeeze(1)

        table_num_logit = self.dense4(
            self.dropout(
                nn.Tanh()(
                    self.dense3(v_T_)
                )
            )
        )

        # num_mask = d_mask[:, :Data.MAX_NUM["table_num"]
        # if num_mask.size(-1) < Data.MAX_NUM["table_num"]:
        #     num_mask = torch.cat([num_mask
        #                              ,
        #                           torch.zeros((num_mask.size(0), Data.MAX_NUM["table_num"] - num_mask.size(-1))).to(
        #                               d_mask.device).long()
        #                           ], axis=-1)
        num_mask = d_mask[:, :self.config.max_num["table_num"]]
        if num_mask.size(-1) < self.config.max_num["table_num"]:
            num_mask = torch.cat([num_mask
                                     ,
                                  torch.zeros((num_mask.size(0), self.config.max_num["table_num"] - num_mask.size(-1))).to(
                                      d_mask.device).long()
                                  ], axis=-1)

        table_num_logit = table_num_logit.masked_fill(num_mask.long() == 0, -1e9)
        if labels is not None:
            loss += self.ce_loss(table_num_logit, table_num_label.long())
        preds += ((
                      to_cpu(nn.Softmax(dim=-1)(table_num_logit))
                      , None if labels is None else to_cpu(table_num_label)
                  ),)

        return loss, preds


class GenClause(nn.Module):
    """
    Common network structure for following clauses :
    "select", "orderby", "groupby", "where", "having"
    """

    def __init__(self, config, clause_type="select"):
        super().__init__()
        self.clause_type = clause_type
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.dense3 = nn.Linear(config.hidden_size, Data.MAX_NUM[clause_type], bias=False)
        self.dense3 = nn.Linear(config.hidden_size, config.max_num[clause_type], bias=False)
        self.dense4 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense5 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense6 = nn.Linear(config.hidden_size, 1, bias=False)
        self.dense7 = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)

        # self.num = nn.Linear(config.hidden_size, Data.MAX_NUM[clause_type])
        self.num = nn.Linear(config.hidden_size, config.max_num[clause_type])
        if clause_type != "groupby":
            self.dist_1 = nn.Linear(config.hidden_size, 2)  # $DIST_1
            self.dist_2 = nn.Linear(config.hidden_size, 2)  # $DIST_2
            self.agg_1 = nn.Linear(config.hidden_size, Data.AGG_NUM)  # $AGG_1
            self.agg_2 = nn.Linear(config.hidden_size, Data.AGG_NUM)  # $AGG_2
            self.ari = nn.Linear(config.hidden_size, Data.ARI_NUM)  # $ARI
            if clause_type == "select":
                self.dist = nn.Linear(config.hidden_size, 2)
                self.agg = nn.Linear(config.hidden_size, Data.AGG_NUM)
            elif clause_type == "orderby":
                self.sort = nn.Linear(config.hidden_size, 2)
            elif clause_type == "where" or clause_type == "having":
                self.conj = nn.Linear(config.hidden_size, len(Data.CONJ_MAP))
                self.not_cond = nn.Linear(config.hidden_size, 2)
                self.cond = nn.Linear(config.hidden_size, Data.OPS_NUM)
                self.nest_1 = nn.Linear(config.hidden_size, 2)
                self.nest_2 = nn.Linear(config.hidden_size, 2)
            else:
                raise Exception("invalid clause type ({})!".format(clause_type))

        self.clause_type = clause_type
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=Data.ignore_idx
        )
        # self.max_num = Data.MAX_NUM[clause_type]
        self.max_num = config.max_num[clause_type]
        # print(clause_type, self.max_num, config.max_num[clause_type])
        self.clause_type = clause_type
        self.config = config

    def forward(self, V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, labels=None):
        batch_size = V_Q.size(0)
        loss = 0

        d1 = self.dense1(V_Q)
        d2 = self.dense2(v_P)
        d3 = d1 + d2.unsqueeze(1)
        d3 = self.dropout(nn.Tanh()(d3))

        A_Q = self.dense3(d3)
        A_Q = A_Q.transpose(-2, -1)

        V_Q_ = torch.matmul(self.softmax(A_Q), V_Q)

        d4 = self.dense4(V_Q_)
        d5 = self.dense5(V_C)

        d6 = d4.unsqueeze(2) + tile(d5, 1, self.max_num)
        d6 = self.dropout(nn.Tanh()(d6))

        A_C_1 = self.dense6(d6).squeeze(-1)
        A_C_1 = A_C_1.masked_fill(c_mask == 0, -1e9)  # maximum column number * total column number

        P_col_1 = self.softmax(A_C_1)

        U_col_C = torch.matmul(P_col_1, V_C)
        U_col_Q_1 = self.dense7(transform(V_Q_, U_col_C))
        U_col_Q_1 = self.dropout(U_col_Q_1)

        m_size = batch_size * self.max_num

        d4 = self.dense4(U_col_Q_1)
        d6 = d4.unsqueeze(2) + tile(d5, 1, self.max_num)
        d6 = self.dropout(nn.Tanh()(d6))

        A_C_2 = self.dense6(d6).squeeze(-1)
        A_C_2 = A_C_2.masked_fill(c_mask == 0, -1e9)
        P_col_2 = self.softmax(A_C_2)

        U_col_C = torch.matmul(P_col_2, V_C)
        U_col_Q_2 = self.dense7(transform(V_Q_, U_col_C))
        U_col_Q_2 = self.dropout(U_col_Q_2)

        preds = tuple()
        num = self.num(v_S)

        if self.clause_type == "select":
            dist = self.dist(v_S)
            agg = self.agg(U_col_Q_1)
            dist_1 = self.dist_1(U_col_Q_1)
            agg_1 = self.agg_1(U_col_Q_1)
            dist_2 = self.dist_2(U_col_Q_2)
            agg_2 = self.agg_2(U_col_Q_2)
            ari = self.ari(U_col_Q_1)

            if labels is not None:
                l_dist, l_num, l_agg, l_unit, l_con1, l_con2 = labels
                l_agg_1, l_col_1, l_dist_1 = torch.split(l_con1, 1, dim=-1)
                l_agg_2, l_col_2, l_dist_2 = torch.split(l_con2, 1, dim=-1)

                loss += self.ce_loss(num, l_num.long())
                loss += self.ce_loss(dist, l_dist.long())

                loss += self.ce_loss(agg.view(m_size, -1), l_agg.view(-1).long())
                loss += self.ce_loss(ari.view(m_size, -1), l_unit.long().view(-1))

                loss += self.ce_loss(agg_1.view(m_size, -1), l_agg_1.view(-1).long())
                loss += self.ce_loss(A_C_1.view(m_size, -1), l_col_1.view(-1).long())
                loss += self.ce_loss(dist_1.view(m_size, -1), l_dist_1.view(-1).long())

                loss += self.ce_loss(agg_2.view(m_size, -1), l_agg_2.view(-1).long())
                loss += self.ce_loss(A_C_2.view(m_size, -1), l_col_2.view(-1).long())
                loss += self.ce_loss(dist_2.view(m_size, -1), l_dist_2.view(-1).long())

            preds += ((to_cpu(self.softmax(dist)), None if labels is None else to_cpu(l_dist)),)
            preds += ((to_cpu(self.softmax(num)), None if labels is None else to_cpu(l_num)),)
            preds += ((to_cpu(self.softmax(agg)), None if labels is None else to_cpu(l_agg)),)
            preds += ((to_cpu(self.softmax(ari)), None if labels is None else to_cpu(l_unit)),)
            preds += ((to_cpu(self.softmax(agg_1)), None if labels is None else to_cpu(l_agg_1)),)
            preds += ((to_cpu(self.softmax(A_C_1)), None if labels is None else to_cpu(l_col_1)),)
            preds += ((to_cpu(self.softmax(dist_1)), None if labels is None else to_cpu(l_dist_1)),)
            preds += ((to_cpu(self.softmax(agg_2)), None if labels is None else to_cpu(l_agg_2)),)
            preds += ((to_cpu(self.softmax(A_C_2)), None if labels is None else to_cpu(l_col_2)),)
            preds += ((to_cpu(self.softmax(dist_2)), None if labels is None else to_cpu(l_dist_2)),)

        elif self.clause_type == "orderby":
            sort = self.sort(v_S)
            dist_1 = self.dist_1(U_col_Q_1)
            agg_1 = self.agg_1(U_col_Q_1)
            dist_2 = self.dist_2(U_col_Q_2)
            agg_2 = self.agg_2(U_col_Q_2)
            ari = self.ari(U_col_Q_1)

            if labels is not None:
                l_sort, l_num, l_ari, l_con1, l_con2 = labels
                l_agg_1, l_col_1, l_dist_1 = torch.split(l_con1, 1, dim=-1)
                l_agg_2, l_col_2, l_dist_2 = torch.split(l_con2, 1, dim=-1)

                loss += self.ce_loss(num, l_num.long())
                loss += self.ce_loss(sort, l_sort.long())
                loss += self.ce_loss(
                    ari.view(-1, ari.size(-1))
                    , l_ari.long().view(-1)
                )

                loss += self.ce_loss(agg_1.view(m_size, -1), l_agg_1.view(-1).long())
                loss += self.ce_loss(A_C_1.view(m_size, -1), l_col_1.view(-1).long())
                loss += self.ce_loss(dist_1.view(m_size, -1), l_dist_1.view(-1).long())

                loss += self.ce_loss(agg_2.view(m_size, -1), l_agg_2.view(-1).long())
                loss += self.ce_loss(A_C_2.view(m_size, -1), l_col_2.view(-1).long())
                loss += self.ce_loss(dist_2.view(m_size, -1), l_dist_2.view(-1).long())

            preds += ((to_cpu(self.softmax(sort)), None if labels is None else to_cpu(l_sort)),)
            preds += ((to_cpu(self.softmax(num)), None if labels is None else to_cpu(l_num)),)
            preds += ((to_cpu(self.softmax(ari)), None if labels is None else to_cpu(l_ari)),)
            preds += ((to_cpu(self.softmax(agg_1)), None if labels is None else to_cpu(l_agg_1)),)
            preds += ((to_cpu(self.softmax(A_C_1)), None if labels is None else to_cpu(l_col_1)),)
            preds += ((to_cpu(self.softmax(dist_1)), None if labels is None else to_cpu(l_dist_1)),)
            preds += ((to_cpu(self.softmax(agg_2)), None if labels is None else to_cpu(l_agg_2)),)
            preds += ((to_cpu(self.softmax(A_C_2)), None if labels is None else to_cpu(l_col_2)),)
            preds += ((to_cpu(self.softmax(dist_2)), None if labels is None else to_cpu(l_dist_2)),)

        elif self.clause_type == "groupby":
            if labels is not None:
                l_num, l_col = labels
                _, l_col, _ = torch.split(l_col, 1, dim=-1)
                l_col = l_col.squeeze(-1)
                loss += self.ce_loss(num, l_num.long())
                loss += self.ce_loss(A_C_1.view(m_size, -1), l_col.view(-1).long())

            preds += ((to_cpu(self.softmax(num)), None if labels is None else to_cpu(l_num)),)
            preds += ((to_cpu(self.softmax(A_C_1)), None if labels is None else to_cpu(l_col)),)

        else:
            dist_1 = self.dist_1(U_col_Q_1)  # $dist_1
            agg_1 = self.agg_1(U_col_Q_1)  # $agg_1
            dist_2 = self.dist_2(U_col_Q_2)  # $dist_2
            agg_2 = self.agg_2(U_col_Q_2)  # $agg_2
            ari = self.ari(U_col_Q_1)  # $ari

            conj = self.conj(U_col_Q_1)
            not_cond = self.not_cond(U_col_Q_1)
            cond = self.cond(U_col_Q_1)
            nest_1 = self.nest_1(U_col_Q_1)
            nest_2 = self.nest_2(U_col_Q_2)

            if labels is not None:
                l_num, l_op, l_ari, l_con1, l_con2, l_conj, l_val, _ = labels
                l_not_op, l_op = torch.split(l_op, 1, dim=-1)
                l_agg_1, l_col_1, l_dist_1 = torch.split(l_con1, 1, dim=-1)
                l_agg_2, l_col_2, l_dist_2 = torch.split(l_con2, 1, dim=-1)
                l_nest_1, l_nest_2 = torch.split(l_val, 1, dim=-1)

                loss += self.ce_loss(num, l_num.long())

                loss += self.ce_loss(agg_1.view(m_size, -1), l_agg_1.view(-1).long())
                loss += self.ce_loss(A_C_1.view(m_size, -1), l_col_1.view(-1).long())
                loss += self.ce_loss(dist_1.view(m_size, -1), l_dist_1.view(-1).long())

                loss += self.ce_loss(agg_2.view(m_size, -1), l_agg_2.view(-1).long())
                loss += self.ce_loss(A_C_2.view(m_size, -1), l_col_2.view(-1).long())
                loss += self.ce_loss(dist_2.view(m_size, -1), l_dist_2.view(-1).long())

                loss += self.ce_loss(conj.view(m_size, -1), l_conj.view(-1).long())
                loss += self.ce_loss(not_cond.view(m_size, -1), l_not_op.view(-1).long())

                loss += self.ce_loss(cond.view(m_size, -1), l_op.view(-1).long())
                loss += self.ce_loss(nest_1.view(m_size, -1), l_nest_1.view(-1).long())
                loss += self.ce_loss(nest_2.view(m_size, -1), l_nest_2.view(-1).long())
                loss += self.ce_loss(ari.view(m_size, -1), l_ari.long().view(-1))

            preds += ((to_cpu(self.softmax(num)), None if labels is None else to_cpu(l_num)),)

            preds += ((to_cpu(self.softmax(conj)), None if labels is None else to_cpu(l_conj)),)
            preds += ((to_cpu(self.softmax(not_cond)), None if labels is None else to_cpu(l_not_op)),)
            preds += ((to_cpu(self.softmax(cond)), None if labels is None else to_cpu(l_op)),)
            preds += ((to_cpu(self.softmax(nest_1)), None if labels is None else to_cpu(l_nest_1)),)
            preds += ((to_cpu(self.softmax(nest_2)), None if labels is None else to_cpu(l_nest_2)),)
            preds += ((to_cpu(self.softmax(ari)), None if labels is None else to_cpu(l_ari)),)

            preds += ((to_cpu(self.softmax(agg_1)), None if labels is None else to_cpu(l_agg_1)),)
            preds += ((to_cpu(self.softmax(A_C_1)), None if labels is None else to_cpu(l_col_1)),)
            preds += ((to_cpu(self.softmax(dist_1)), None if labels is None else to_cpu(l_dist_1)),)
            preds += ((to_cpu(self.softmax(agg_2)), None if labels is None else to_cpu(l_agg_2)),)
            preds += ((to_cpu(self.softmax(A_C_2)), None if labels is None else to_cpu(l_col_2)),)
            preds += ((to_cpu(self.softmax(dist_2)), None if labels is None else to_cpu(l_dist_2)),)

        return loss, preds


class LimitClause(nn.Module):
    "Network for limit clause"

    def __init__(self, config):
        super().__init__()
        self.top1 = nn.Linear(config.hidden_size, 2)
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=Data.ignore_idx)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, V_Q, v_P, v_S, q_mask, labels=None):
        loss = 0
        top1 = self.top1(v_S)
        d1 = self.dense1(V_Q)
        d2 = self.dense2(v_P)
        d3 = self.dense3(nn.Tanh()(d1 + d2.unsqueeze(1))).squeeze(-1)

        d3 = d3.masked_fill(q_mask == 0, -1e9)

        if labels is not None:
            l_top1, l_pos = labels
            loss += self.ce_loss(top1, l_top1.view(-1).long())
            loss += self.ce_loss(d3, l_pos.view(-1).long())
        preds = (
            (to_cpu(self.softmax(top1)), None if labels is None else to_cpu(l_top1))
            , (to_cpu(self.softmax(d3)), None if labels is None else to_cpu(l_pos))
        )
        return loss, preds


class Model(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        model_config = transformers.AutoConfig.from_pretrained(config.pretrained_model)
        self.embedding = Embedding(config, encoder)
        self.spc_embedding = nn.Embedding(len(Data.SPC), model_config.hidden_size)

        self.q_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        self.t_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        self.d_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        self.p_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        # self.q_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.t_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.d_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.p_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # if "electra" in config.pretrained_model:
        #     print("\n==== Load Electra self Attention ====\n")
        #     self.q_self_attn = transformers.modeling_electra.ElectraSelfAttention(model_config)
        #     self.t_self_attn = transformers.modeling_electra.ElectraSelfAttention(model_config)
        #     self.d_self_attn = transformers.modeling_electra.ElectraSelfAttention(model_config)
        #     self.p_self_attn = transformers.modeling_electra.ElectraSelfAttention(model_config)
        # else:
        #     self.q_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        #     self.t_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        #     self.d_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        #     self.p_self_attn = transformers.modeling_bert.BertSelfAttention(model_config)
        # self.q_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.t_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.d_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        # self.p_self_attn = transformers.models.bert.modeling_bert.BertSelfAttention(model_config)
        self.dense_s = nn.Linear(5 * config.hidden_size, config.hidden_size)

        self.clause_layers = nn.ModuleList([])
        # intersection, union, except, non
        self.ex_pred_name = Data.EX_LIST
        # add clause existance binary prediction layers
        # bg, bo, bl, bw, bh
        for _ in range(len(self.ex_pred_name) - 1):
            self.clause_layers.append(
                nn.Linear(config.hidden_size, 2)
            )
        # add iuen pred layer
        self.clause_layers.append(nn.Linear(config.hidden_size, 4))

        self.gen_tbl = TableClause(config)
        self.gen_sel = GenClause(config, clause_type="select")
        self.gen_ord = GenClause(config, clause_type="orderby")
        self.gen_grb = GenClause(config, clause_type="groupby")
        self.gen_lim = LimitClause(config)
        self.gen_whe = GenClause(config, clause_type="where")
        self.gen_hav = GenClause(config, clause_type="having")

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=Data.ignore_idx)
        self.config = config

    def encode(self, x, mask, q_len, c_len, t_len):
        batch_size = x.size(0)

        # get input encodes
        out = self.embedding(x, mask)
        _, V_Q, V_C, V_T, q_mask, c_mask, t_mask, d_mask = extract_vectors(
            self.config
            , out
            , q_len
            , c_len
            , t_len
        )

        # self-attn query encode
        v_Q = self.q_self_attn(V_Q, q_mask.unsqueeze(1).unsqueeze(1))[0][:, 0]

        # self-attn column vector==table encode
        v_T = self.t_self_attn(V_T.view(-1, V_T.size(-2), V_T.size(-1))
                               , t_mask.unsqueeze(1).unsqueeze(1)
                               )[0][:, 0]
        v_T = v_T.view(batch_size, -1, v_T.size(-1))

        # DB schema encode
        v_D = self.d_self_attn(v_T, d_mask.unsqueeze(1).unsqueeze(1))[0][:, 0]

        return V_Q, V_C, q_mask, c_mask, d_mask, v_Q, v_T, v_D

    def gen_sql(self, V_Q, V_C, v_Q, v_T, v_D, q_mask, c_mask, d_mask, labels):
        loss = 0
        li = utils_ko.IncrementIndex(max_num=len(labels))
        preds = dict()

        # SPC id encode
        spc_id = labels[li.get()]
        spc_mask = labels[li.get()]
        v_P = self.spc_embedding(spc_id.long())
        v_P = self.p_self_attn(v_P, spc_mask.long().unsqueeze(1).unsqueeze(1)
                               )[0][:, 0]

        # combine encodes
        v_S = self.dense_s(torch.cat([transform(v_Q, v_D), v_P], dim=-1))

        # do clause existance predictions
        for i in range(len(self.ex_pred_name)):
            logit = self.clause_layers[i](v_S)
            if labels is not None:
                clause_labels = labels[li.get()].long()
                loss += self.ce_loss(logit, clause_labels)
            preds[self.ex_pred_name[i]] = (
                to_cpu(nn.Softmax(dim=-1)(logit))
                , None if labels is None else to_cpu(clause_labels)
                ,)

        # get table labels
        # table_num, table_ids
        tbl_loss, pred = self.gen_tbl(
            v_T, v_Q, v_D, v_P, d_mask
            , labels=(labels[li.get()], labels[li.get()])
        )
        preds["table"] = pred
        loss += tbl_loss

        # get select clause labels
        # cond_num, dist, agg, ari
        # , (dist_1, agg_1, col_1)
        # , (dist_2, agg_2, col_2)
        sel_labels = tuple([labels[li.get()] for _ in range(6)])
        # """
        sel_loss, pred = self.gen_sel(V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, sel_labels)
        preds["select"] = pred
        loss += sel_loss
        # """

        # get orderby clause labels
        # cond_num, sort, ari
        # , (dist_1, agg_1, col_1)
        # , (dist_2, agg_2, col_2)
        ord_labels = tuple([labels[li.get()] for _ in range(5)])
        # """
        ord_loss, pred = self.gen_ord(V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, ord_labels)
        preds["orderby"] = pred
        loss += ord_loss
        # """

        # get groupby clause labels
        # cond_num, col
        grb_labels = tuple([labels[li.get()] for _ in range(2)])
        # """
        grb_loss, pred = self.gen_grb(V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, grb_labels)
        preds["groupby"] = pred
        loss += grb_loss
        # """

        # get limit clause labels
        # is_top1, val_pos
        lim_labels = tuple([labels[li.get()] for _ in range(2)])
        # """
        lim_loss, pred = self.gen_lim(V_Q, v_P, v_S, q_mask, lim_labels)
        preds["limit"] = pred
        loss += lim_loss
        # """

        # get where clause labels
        whe_labels = tuple([labels[li.get()] for _ in range(8)])
        # """
        whe_loss, pred = self.gen_whe(V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, whe_labels)
        preds["where"] = pred
        loss += whe_loss
        # """

        # get having clause labels
        hav_labels = tuple([labels[li.get()] for _ in range(8)])
        # """
        hav_loss, pred = self.gen_hav(V_Q, V_C, v_Q, v_P, v_D, v_S, q_mask, c_mask, hav_labels)
        preds["having"] = pred
        loss += hav_loss
        # """

        return loss, preds

    def forward(self, x, mask, q_len, c_len, t_len, sql_mask, all_labels=None, table_map=None, train=True,
                tables=None, utt_ids=None
                ):
        batch_size = x.size(0)

        V_Q, V_C, q_mask, c_mask, d_mask, v_Q, v_T, v_D = \
            self.encode(x, mask, q_len, c_len, t_len)

        max_sql_num = len(all_labels)

        def idx_tensor(t, idx):
            return t[idx == 1]

        results = dict()
        loss = 0
        decode_results = [[] for _ in range(batch_size)]

        for i in range(max_sql_num):
            idx = sql_mask[:, i].view(-1)
            labels = all_labels[i]
            indexed_labels = tuple()

            for l in labels:
                # print(l)
                # print(idx)
                # print()
                indexed_labels += (idx_tensor(l, idx),)
            # exit(-1)
            depth_loss, preds = self.gen_sql(
                idx_tensor(V_Q, idx)
                , idx_tensor(V_C, idx)
                , idx_tensor(v_Q, idx)
                , idx_tensor(v_T, idx)
                , idx_tensor(v_D, idx)
                , idx_tensor(q_mask, idx)
                , idx_tensor(c_mask.view(batch_size, 1, -1), idx)
                , idx_tensor(d_mask, idx)
                , indexed_labels
            )

            loss += depth_loss
            idx = idx.cpu().numpy()
            valid_idx = np.where(idx != 0)[0]

            if not train:
                """
                    set target for flexible sql len (depth) in mini-batch 
                """
                target_tables = []
                target_utt_ids = []

                if utt_ids:
                    """
                        if decode sql during evaluate
                    """
                    for batch_index in range(batch_size):
                        if idx[batch_index]:
                            target_tables.append(tables[batch_index])
                            target_utt_ids.append(utt_ids[batch_index])

                # result, decode_sql_list = utils_ko.check_pred(
                #     np.sum(idx).item()
                #     , preds
                #     , table_map[valid_idx]
                #     , tables
                #     , utt_ids
                # )
                result, decode_sql_list = utils_ko.check_pred(
                    np.sum(idx).item()
                    , preds
                    , table_map[valid_idx]
                    , target_tables
                    , target_utt_ids
                )

                # result = utils.check_pred(
                #     np.sum(idx).item()
                #     , preds
                #     , table_map[valid_idx]
                #     , tables
                #     , utt_ids
                # )
                # for batch_index, decode_sql in enumerate(decode_sql_list):
                #     decode_results[batch_index].append(decode_sql)

                if decode_sql_list:
                    """
                        evaluation process
                    """
                    for batch_index in range(batch_size):
                        if idx[batch_index]:
                            decode_results[batch_index].append(decode_sql_list.pop(0))

                if i == 0:
                    results = result
                    results["final_result"] = np.copy(result["final_sample"])
                else:
                    results["final_result"][valid_idx] *= result["final_sample"]

        loss /= max_sql_num

        return loss, results, utils_ko.concat_decode_sql(decode_results, utt_ids)
