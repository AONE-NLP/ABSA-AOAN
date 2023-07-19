# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class AOAN(nn.Module):
    def __init__(self, bert, opt):
        super(AOAN, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.pool=nn.AvgPool1d(opt.SRD+1)


    def moving_mask(self, text_local_indices, aspect_indices, mask_len):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, _ = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids,return_dict=False)
        bert_spc_out = self.dropout(bert_spc_out)

        neighboring_span, _ = self.bert_local(text_local_indices,return_dict=False)
        neighboring_span = self.dropout(neighboring_span)

        out_list=[]

        for i in range(self.opt.threshold+1):
            masked_local_text_vec = self.moving_mask(text_local_indices, aspect_indices,i)
            neighboring_span = torch.mul(neighboring_span, masked_local_text_vec)
            enhanced_text = torch.cat((neighboring_span, bert_spc_out), dim=-1)
            mean_pool = self.linear_double(enhanced_text)
            self_attention_out= self.bert_SA(mean_pool)
            pooled_out = self.bert_pooler(self_attention_out)
            dense_out = self.dense(pooled_out)
            out_list.append(dense_out )
        out=torch.cat(out_list,dim=-1)
        out=out.view(dense_out.shape[0],3,-1)

        ensem_out=self.pool(out)

        ensem_out=ensem_out.squeeze(-1)

        return ensem_out
