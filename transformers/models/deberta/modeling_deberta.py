# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeBERTa model. """

import math
from collections.abc import Sequence

import torch
from packaging import version
from torch import _softmax_backward_data, nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_deberta import DebertaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
]


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
      dim (int): The dimension that will apply softmax

    Example::
      import torch
      from transformers.models.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
        else:
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).byte()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:

        drop_prob (float): the dropout probabilities

    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout


        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_type)
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

import numpy as np
import scipy.sparse as sp
from torch.nn import init

def normalized_adj_torch_NI(A):
    A_hat = A - torch.diag(A) + torch.eye(A.shape[0]).to(A.device)
    D = A_hat.sum(-1).reshape(-1,)
    D[D <= 1e-4] = 1e-4
    diag = torch.reciprocal(torch.sqrt(D))
    A_wave = torch.mul(torch.mul(diag.reshape(-1, 1), A_hat), diag.reshape(1,-1))
    return A_wave

# @profile
def get_attn_adj(scores, dim=-1, Atype=0, key=None, super_p=None, m_top=None):
    A2 = scores.detach()
    A2 = torch.where(torch.isnan(A2), torch.full_like(A2, 0), A2)
    bs, he, lq, lk = A2.shape
    len_key = torch.max(torch.norm(key, dim = -1, keepdim=True), torch.ones((1,1,1,1)).to(key.device))
    len_key = len_key.unsqueeze(-1)
    Ak2 = torch.einsum("bhyk,bhzk->bhkyz", A2, A2) / len_key
    Ak2 = Ak2 / key.shape[-1]
    index_1 = Ak2 >= super_p
    Ak2[Ak2 < super_p] = 0
    Ak2[index_1] = 1
    A2 = torch.mean(Ak2[:,:,:,:,:], dim=2, keepdim=False)

    # ## ori_AK1
    # get_adj = normalized_adj_torch_NI
    # Ak1 = torch.zeros_like(A2)
    # for b in range(bs):
    #     for h in range(he):
    #         Ak1[b,h] = get_adj(A2[b,h])
    # return Ak1.detach()

    ## new_AK1
    diag_ = torch.arange(0, lk)
    A2[torch.arange(bs)[:, None, None],
        torch.arange(he)[None, :, None],
        torch.arange(lq)[None, None, :],
        diag_] = 1 #A_hat
    D = A2.sum(-1)
    D[D <= 1e-4] = 1e-4
    diag = torch.reciprocal(torch.sqrt(D)).unsqueeze(-1)
    A_wave = torch.mul(torch.mul(diag, A2), diag.transpose(-1, -2))
    # print((A_wave != Ak1).sum())
    return A_wave
    

import scipy.sparse as sp
from torch.nn import init

class CheckerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.order = config.order or 2
        self.heads = config.num_attention_heads
        self.oheads = config.oheads
        self.octrl = config.octrl
        self.Atype = config.Atype
        self.checkertype = config.checkertype or 0
        self.max_len = config.max_position_embeddings
        self.use = config.num_attention_heads - self.oheads
        ofactor = 0 if self.octrl==0 else 1
        self.orders = self.order*1+ofactor if self.checkertype==0 else 1+ofactor
        dim = self.max_len; c_in = self.orders*dim

        self.W = nn.Parameter(torch.Tensor(self.max_len, c_in), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(self.max_len), requires_grad=True)
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.super_p = config.super_p
        self.need_W = config.need_W
        self.prob_attn = config.prob_attn
        self.factor = config.factor
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    # @profile
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        # M_top, _ = torch.sort(M_top, descending=False, dim=-1)
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        K_reduce = K[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K_reduce.transpose(-2, -1)) # factor*ln(L_q)*L_k
        return Q_K, M_top, Q_reduce, K_reduce
    
    # @profile
    def _merge(self, A, M_top, L_q, full_x):
        b, h, x, y = A.shape
        # index = torch.zeros((b, h, x, L_q)).to("cuda")
        # index[torch.arange(b)[:, None, None],
        #       torch.arange(h)[None, :, None],
        #       torch.arange(x)[None, None, :],
        #       M_top] = 1
        # index2 = torch.zeros((b, h, L_q)).to("cuda")
        # index2[torch.arange(b)[:, None, None],
        #       torch.arange(h)[None, :, None],
        #       M_top] = 1
        # Index = torch.einsum("bhx,bhy->bhxy", index2, index2).to("cuda")
        # resB = torch.matmul(A.transpose(-1, -2), index)
        # resB = torch.matmul(resB.transpose(-1, -2), index)
        # resB = resB + (Index == 0) * full_x
        resC = full_x
        resC[torch.arange(b)[:, None, None, None],
              torch.arange(h)[None, :, None, None],
              M_top.unsqueeze(-1),
              M_top.unsqueeze(-2)] = A
        # print((resB != resC).sum())
        # exit(0)
        return resC
    
    # @profile
    def forward(self, x, query, key):
        # x [B, H, L, L]
        if self.use <= 6:
            x = x[:,0:self.use*2:2,:,:].contiguous()
        else:
            x = x[:,0:self.use,:,:].contiguous()
        full_x = x
        B, H, Lk, L = x.shape
        if self.prob_attn:
            _, _, L_K, _ = key.shape
            _, _, L_Q, _ = query.shape
            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
            U_part = U_part if U_part<L_K else L_K
            u = u if u<L_Q else L_Q
            x, m_top, Q_reduce, K_reduce = self._prob_QK(query, key, sample_k=U_part, n_top=u)
        
        if self.prob_attn:
            A1 = get_attn_adj(x, -1, self.Atype, K_reduce, self.super_p)
        else:
            A1 = get_attn_adj(x, -1, self.Atype, key, self.super_p)
        # if self.prob_attn:
        #     A1 = self._merge(A1, m_top, Lk)
        
        out = [] if self.octrl==0 else [x]
        Ws = self.W.chunk(self.orders, 1)
        if self.checkertype==0:
            for A in [A1]:
                if not self.need_W:
                    #this
                    if self.prob_attn:
                        x1 = torch.matmul(A, Q_reduce)
                        x1 = torch.matmul(x1, K_reduce.transpose(-1, -2))
                        x1 = torch.matmul(x1, A.transpose(-1, -2))
                        out.append(self._merge(x1, m_top, Lk, full_x))
                        for i in range(1, self.orders):
                            x1 = torch.matmul(A, x1)
                            x1 = torch.matmul(x1, A.transpose(-1, -2))
                            out.append(self._merge(x1, m_top, Lk, full_x))
                    else:
                        x1 = torch.matmul(A, query)
                        x1 = torch.matmul(x1, key.transpose(-1, -2))
                        x1 = torch.matmul(x1, A.transpose(-1, -2))
                        out.append(x1)
                        for i in range(1, self.orders):
                            x1 = torch.matmul(A, x1)
                            x1 = torch.matmul(x1, A.transpose(-1, -2))
                            out.append(x1)
                else:
                    exit(0)
                    for i in range(self.orders):
                        x1 = torch.matmul(A, x).contiguous()
                        x1 = x1.matmul(Ws[i][:,:L].t()) + self.b
                        out.append(x1)
                        x = x1
        else:
            print("error")
            exit(0)
        h = torch.Tensor([0.]).to("cuda")
        for o in out:
            h = h + o[:,:,:Lk,:L]
        h = self.dropout(h)
        return h.contiguous()

class DebertaAttention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.self = DisentangledSelfAttention(config, layer)
        self.output = DebertaSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaOutput(nn.Module):
    def __init__(self, config):
        super(DebertaOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaLayer(nn.Module):
    def __init__(self, config, layer):
        super(DebertaLayer, self).__init__()
        self.attention = DebertaAttention(config, layer)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_att=return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output


class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaLayer(config, _) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(
                next_kv,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def build_relative_position(query_size, key_size, device):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledSelfAttention(torch.nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`

    """

    def __init__(self, config, layer):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = torch.nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = torch.nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = torch.nn.Linear(
                config.num_attention_heads, config.num_attention_heads, bias=False
            )

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_proj = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_q_proj = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

        self.layer = layer
        self.olayers = [int(l) for l in config.olayers.replace(' ','').split(',')]
        if self.layer in self.olayers:
            self.checkernet = CheckerNet(config)
        
        self.oheads = config.oheads
        self.otype = config.otype
        self.ceof = config.ceof
        self.use_hattn = config.use_hattn
        self.attn_heads = config.attn_heads
        self.easy_hattn = config.easy_hattn

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_hattn and (self.layer in self.olayers):
            use = self.num_attention_heads - self.oheads
            if use <= 6:
                h_attn = self.checkernet(attention_scores, query_layer[:,0:use*2:2,:,:], key_layer[:,0:use*2:2,:,:])
            else:
                h_attn = self.checkernet(attention_scores, query_layer[:,0:use,:,:], key_layer[:,0:use,:,:])
            if self.otype==0:
                #use = self.num_attention_heads - self.oheads
                if not self.easy_hattn:
                    bb, hh, qq, kk = attention_scores.shape
                    h_index = torch.tensor(self.attn_heads)
                    attention_scores[torch.arange(bb)[:, None, None, None],
                                    h_index[None, :, None, None],
                                    torch.arange(qq)[None, None, :, None],
                                    torch.arange(kk)[None, None, None, :]] = attention_scores[torch.arange(bb)[:, None, None, None],
                                                                                h_index[None, :, None, None],
                                                                                torch.arange(qq)[None, None, :, None],
                                                                                torch.arange(kk)[None, None, None, :]] * (1 - self.ceof) + h_attn * self.ceof
                else:
                    if use <= 6:
                        attention_scores[:,0:use*2:2,:,:] = attention_scores[:,0:use*2:2,:,:] * (1 - self.ceof) + h_attn * self.ceof
                    else:
                        attention_scores[:,0:use,:,:] = attention_scores[:,0:use,:,:] * (1 - self.ceof) + h_attn * self.ceof
                # attention_scores[:,self.oheads:,:,:] = attention_scores[:,self.oheads:,:,:] * (1 - self.ceof) + h_attn * self.ceof
            elif self.otype==1:
                attention_scores[:,self.oheads:,:,:] = h_attn

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)
        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att

        return score


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DebertaConfig
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```


    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DebertaTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/deberta-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = torch.nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = torch.nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
