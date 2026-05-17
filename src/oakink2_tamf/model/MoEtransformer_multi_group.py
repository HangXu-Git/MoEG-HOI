import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch.nn as nn
import math


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_topk_idx = encoder_layer.return_topk_idx
        self.need_weights = encoder_layer.need_weights

    def forward(self, src: Tensor, cond_router: Tensor, emb_refine_text: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        all_attention_weights = []
        all_topk_idx = []
        all_topk_weight = []
        all_scores = []
        aux_loss_list = []
        aux_loss_t_list = []
        aux_loss_a_list = []
        expert_id_list = [-1, -1, -1, -1, -1, -1, -1, -1]
        i = 0

        for mod in self.layers:
            expert_id = expert_id_list[i]
            if self.return_topk_idx:
                output, aux_loss, attention_weight, topk_idx, topk_weight, scores = mod(output, cond_router, emb_refine_text, src_mask=mask, src_key_padding_mask=src_key_padding_mask, selected_expert_id=expert_id)
                all_topk_idx.append(topk_idx.detach())
                all_topk_weight.append(topk_weight.detach())
                all_scores.append(scores.detach())
            else:
                output, aux_loss, attention_weight = mod(output, cond_router, emb_refine_text, src_mask=mask, src_key_padding_mask=src_key_padding_mask, selected_expert_id=expert_id)

            if attention_weight is not None:
                all_attention_weights.append(attention_weight.detach())

            if aux_loss is not None:
                aux_loss_list.append(aux_loss[0])
                aux_loss_t_list.append(aux_loss[1])
                aux_loss_a_list.append(aux_loss[2])

            i = i + 1

        if self.return_topk_idx:
            all_topk_idx = torch.stack(all_topk_idx, dim=0)
            all_topk_weight = torch.stack(all_topk_weight, dim=0)
            all_scores = torch.stack(all_scores, dim=0)

        if len(all_attention_weights) != 0:
            all_attention_weights = torch.stack(all_attention_weights, dim=0)

        if self.norm is not None:
            output = self.norm(output)
        
        aux_list_tuple = (aux_loss_list, aux_loss_t_list, aux_loss_a_list)
        if self.return_topk_idx:
            return output, aux_list_tuple, all_attention_weights, all_topk_idx, all_topk_weight, all_scores
        else:
            return output, aux_list_tuple, all_attention_weights


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, num_experts: int = 6, 
                 num_shared_experts: Optional[int] = None, num_experts_per_tok: int = 2, num_group: int = 3, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool=False, need_weights: bool=False, return_topk_idx: bool=False,
                 use_refine_text=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.FFN = MultiGroupSparseMoeBlock(d_model, dim_feedforward, num_experts=num_experts, num_shared_experts=num_shared_experts, num_experts_per_tok=num_experts_per_tok, num_group=num_group, batch_first=batch_first, return_topk_idx=return_topk_idx, use_refine_text=use_refine_text)
        self.norm_first = norm_first
        self.need_weights = need_weights
        self.return_topk_idx = return_topk_idx
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, cond_router: Tensor, emb_refine_text: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, selected_expert_id=-1) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            output, attention_weight = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, need_weights=self.need_weights)
            x = x + output
            if self.return_topk_idx:
                x_out, topk_idx, topk_weight, scores = self.FFN(self.norm2(x), cond_router, emb_refine_text, selected_expert_id=selected_expert_id)
                x = x + x_out
            else:
                x = x + self.FFN(self.norm2(x), cond_router, emb_refine_text, selected_expert_id=selected_expert_id)
        else:
            output, attention_weight = self._sa_block(x, src_mask, src_key_padding_mask, need_weights=self.need_weights)
            x = self.norm1(x + output)
            if self.return_topk_idx:
                x_out, topk_idx, topk_weight, scores = self.FFN(x, cond_router, emb_refine_text, selected_expert_id=selected_expert_id)
                x = self.norm2(x + x_out)
            else:
                x = self.norm2(x + self.FFN(x, cond_router, emb_refine_text, selected_expert_id=selected_expert_id))
        
        aux_loss_tuple = self.FFN.get_aux_loss()

        if self.return_topk_idx:
            return x, aux_loss_tuple, attention_weight, topk_idx, topk_weight, scores
        else:
            return x, aux_loss_tuple, attention_weight

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], need_weights: bool = False) -> Tensor:
        
        if need_weights:
            x, attention_weight = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)
            return self.dropout1(x), attention_weight
        else:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)[0]
            return self.dropout1(x), None


# MoE
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=3, num_experts_per_tok=1, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape      
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores    # [bs * seq_len, n_routed_experts]
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)   # [bs, seq_len * topk]
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bs, seq_len, n_routed_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss, scores
    

class MultiGroupMoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=3, num_experts_per_tok=1, num_group=3, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.num_group = num_group

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = True
        self.gating_dim = embed_dim
        self.weight_list = nn.ParameterList()
        for i in range(num_group):
            self.weight_list.append(nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        for weight in self.weight_list:
            init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, hidden_states):  # hidden_states: [bsz, 1, d*2]
        bsz, seq_len, h = hidden_states.shape      
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        topk_weight_list = []
        topk_idx_list = []
        scores_list = []
        all_aux_loss = 0.0
        for i in range(self.num_group):
            logits = F.linear(hidden_states, self.weight_list[i], None)
            if self.scoring_func == 'softmax':
                scores = logits.softmax(dim=-1)
            else:
                raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
            
            ### select top-k experts
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

            ### expert-level computation auxiliary loss
            if self.training and self.alpha > 0.0:
                scores_for_aux = scores    # [bs * seq_len, n_routed_experts * num_group]
                aux_topk = self.top_k
                topk_idx_for_aux_loss = topk_idx.view(bsz, -1)   # [bs, seq_len * topk * num_group]
                if self.seq_aux:  # not used in our paper
                    print("error: seq_aux is not supported in MultiGroupMoEGate")
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bs, seq_len, n_routed_experts]
                    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                    ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                    aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
                else:
                    mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                    ce = mask_ce.float().mean(0)
                    Pi = scores_for_aux.mean(0)
                    fi = ce * self.n_routed_experts
                    aux_loss = (Pi * fi).sum() * self.alpha
            else:
                aux_loss = 0.0

            all_aux_loss = all_aux_loss + aux_loss
            topk_idx = topk_idx + i * self.n_routed_experts  # adjust index for multi-group
            topk_weight_list.append(topk_weight)
            topk_idx_list.append(topk_idx)
            scores_list.append(scores)
        
        # concatenate top-k weights and indices
        topk_weight = torch.cat(topk_weight_list, dim=-1)
        topk_idx = torch.cat(topk_idx_list, dim=-1)
        scores = torch.cat(scores_list, dim=-1)
        
        ### norm gate to sum 1
        if self.num_group > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight, all_aux_loss, scores
    

class MultiGroupMoEGate_split(nn.Module):
    def __init__(self, embed_dim, num_experts=3, num_experts_per_tok=1, num_group=3, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.num_group = num_group

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = True
        self.gating_dim = embed_dim//2
        self.weight_list_t = nn.ParameterList()
        self.weight_list_a = nn.ParameterList()
        for i in range(num_group):
            self.weight_list_t.append(nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))))
            self.weight_list_a.append(nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))))
        self.reset_parameters()
        print('Use MultiGroupMoEGate_split')

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        for weight in self.weight_list_t:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
        for weight in self.weight_list_a:
            init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, hidden_states):  # hidden_states: [bsz, 1, d*2]
        bsz, seq_len, h = hidden_states.shape      
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        hidden_states_t = hidden_states[:, :self.gating_dim]  # timestep feature
        hidden_states_a = hidden_states[:, self.gating_dim:]  # action feature

        topk_weight_list = []
        topk_idx_list = []
        scores_list = []
        all_aux_loss = 0.0
        all_aux_loss_t = 0.0
        all_aux_loss_a = 0.0

        for i in range(self.num_group):
            logits_t = F.linear(hidden_states_t, self.weight_list_t[i], None)
            logits_a = F.linear(hidden_states_a, self.weight_list_a[i], None)
            logits = logits_t + logits_a  # combine the two features
            if self.scoring_func == 'softmax':
                scores_t = logits_t.softmax(dim=-1)
                scores_a = logits_a.softmax(dim=-1)
                scores = logits.softmax(dim=-1)
            else:
                raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
            
            ### select top-k experts
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
            _, topk_idx_t = torch.topk(scores_t, k=self.top_k, dim=-1, sorted=False)  # use for loss
            _, topk_idx_a = torch.topk(scores_a, k=self.top_k, dim=-1, sorted=False)

            ### expert-level computation auxiliary loss
            if self.training and self.alpha > 0.0:
                # timestep loss
                scores_for_aux = scores_t    # [bs * seq_len, n_routed_experts * num_group]
                topk_idx_for_aux_loss = topk_idx_t.view(bsz, -1)   # [bs, seq_len * topk * num_group]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss_t = (Pi * fi).sum() * self.alpha
                # action loss
                scores_for_aux = scores_a    # [bs * seq_len, n_routed_experts * num_group]
                topk_idx_for_aux_loss = topk_idx_a.view(bsz, -1)   # [bs, seq_len * topk * num_group]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss_a = (Pi * fi).sum() * self.alpha
                # combine the two losses
                aux_loss = (aux_loss_t + aux_loss_a) / 2.0 + torch.abs(aux_loss_t - aux_loss_a)  # balance the two losses
            else:
                aux_loss = 0.0
                aux_loss_t = 0.0
                aux_loss_a = 0.0

            all_aux_loss = all_aux_loss + aux_loss
            all_aux_loss_t = all_aux_loss_t + aux_loss_t
            all_aux_loss_a = all_aux_loss_a + aux_loss_a
            topk_idx = topk_idx + i * self.n_routed_experts  # adjust index for multi-group
            topk_weight_list.append(topk_weight)
            topk_idx_list.append(topk_idx)
            scores_list.append(scores)

        
        # concatenate top-k weights and indices
        topk_weight = torch.cat(topk_weight_list, dim=-1)
        topk_idx = torch.cat(topk_idx_list, dim=-1)
        scores = torch.cat(scores_list, dim=-1)
        
        ### norm gate to sum 1
        if self.num_group > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator


        return topk_idx, topk_weight, all_aux_loss, scores, all_aux_loss_t, all_aux_loss_a, 
    

# class MultiGroupMoEGate_v2(nn.Module):
#     def __init__(self, embed_dim, num_experts=3, num_experts_per_tok=1, num_group=3, aux_loss_alpha=0.01):
#         super().__init__()
#         self.top_k = num_experts_per_tok
#         self.n_routed_experts = num_experts
#         self.num_group = num_group

#         self.scoring_func = 'softmax'
#         self.alpha = aux_loss_alpha
#         self.seq_aux = False

#         # topk selection algorithm
#         self.norm_topk_prob = True
#         self.gating_dim = embed_dim
#         self.level_weight = nn.Parameter(torch.empty((self.num_group, self.gating_dim)))
#         self.expert_weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         import torch.nn.init  as init
#         init.kaiming_uniform_(self.level_weight, a=math.sqrt(5))
#         init.kaiming_uniform_(self.expert_weight, a=math.sqrt(5))

#     def forward(self, hidden_states):
#         bsz, seq_len, h = hidden_states.shape      
#         ### compute gating score
#         hidden_states = hidden_states.view(-1, h)
#         level_logits = F.linear(hidden_states, self.level_weight, None)
#         expert_logits = F.linear(hidden_states, self.expert_weight, None)

#         if self.scoring_func == 'softmax':
#             level_scores = level_logits.softmax(dim=-1)
#             expert_scores = expert_logits.softmax(dim=-1)
#         else:
#             raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

#         scores = expert_scores.unsqueeze(-2) * level_scores.unsqueeze(-1)  # [bs * seq_len, num_group, n_routed_experts]
#         dim = scores.shape[0]  # bsz * seq_len
#         scores = scores.view(dim, -1)  # [bs * seq_len, num_group * n_routed_experts]

#         mask = scores > (1/(self.num_group * self.n_routed_experts))  # [bs * seq_len, num_group * n_routed_experts]
#         weight = mask * scores  # [bs * seq_len, num_group * n_routed_experts]
#         weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)  # norm to sum 1
#         topk_idx = torch.where(mask)

#         ### expert-level computation auxiliary loss
#         if self.training and self.alpha > 0.0:
#             scores_for_aux = scores    # [bs * seq_len, n_routed_experts]
#             aux_topk = self.top_k
#             topk_idx_for_aux_loss = topk_idx.view(bsz, -1)   # [bs, seq_len * topk]
#             if self.seq_aux:
#                 print("error: seq_aux is not supported in MultiGroupMoEGate")
#                 scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bs, seq_len, n_routed_experts]
#                 ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
#                 ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
#                 aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
#             else:
#                 mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
#                 ce = mask_ce.float().mean(0)
#                 Pi = scores_for_aux.mean(0)
#                 fi = ce * self.n_routed_experts
#                 aux_loss = (Pi * fi).sum() * self.alpha
#         else:
#             aux_loss = None
#         return topk_idx, topk_weight, aux_loss, scores
    

class MLP_with_cross_att(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1, batch_first=True):
        super().__init__()
        self.multihead_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=batch_first)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(input_dim, eps=1e-5)

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout_layer = nn.ModuleList([Dropout(dropout) for _ in range(num_layers)])
        

    def forward(self, x, emb_text):
        
        emb_text = emb_text.unsqueeze(1)  # [bs, 1, embed_dim]
        x = self.norm(x + self.cross_att_block(x, emb_text))  # [bs, seqlen, embed_dim]

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.dropout_layer[i](x)
        return x
    
    def cross_att_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout(x)
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout_layer = nn.ModuleList([Dropout(dropout) for _ in range(num_layers)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.dropout_layer[i](x)
        return x
    

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MultiGroupSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, d_model, dim_feedforward, num_experts=3, num_shared_experts=None, num_experts_per_tok=1, num_group=3, batch_first=False, return_topk_idx=False, use_refine_text=True):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok

        # same_size
        # self.experts = nn.ModuleList([MLP_with_cross_att(input_dim=d_model, hidden_dim=dim_feedforward, output_dim=d_model, num_layers=2, batch_first=True) for i in range(num_experts * num_group)])
        # different_size
        self.experts = nn.ModuleList()
        for i in range(num_group):
            for j in range(num_experts):
                dim_feedforward_per_expert = int(dim_feedforward/(2 ** (j+1)))
                if use_refine_text:
                    self.experts.append(MLP_with_cross_att(input_dim=d_model, hidden_dim=dim_feedforward_per_expert, output_dim=d_model, num_layers=2, batch_first=True))
                else:
                    self.experts.append(MLP(input_dim=d_model, hidden_dim=dim_feedforward_per_expert, output_dim=d_model, num_layers=2))
                    
        self.use_refine_text = use_refine_text
        if use_refine_text:  # multi-group
            self.gate = MultiGroupMoEGate_split(embed_dim=d_model*2, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, num_group=num_group) # timestep and action
        else:
            self.gate = MoEGate(embed_dim=d_model*2, num_experts=num_experts*num_group, num_experts_per_tok=1)
        self.n_shared_experts = num_shared_experts
        self.num_group = num_group
        self.num_expert = num_experts
        self.batch_first = batch_first
        self.return_topk_idx = return_topk_idx
        self.aux_loss = 0.0
        self.aux_loss_t = 0.0
        self.aux_loss_a = 0.0
        
        if self.n_shared_experts is not None:
            self.shared_experts = nn.ModuleList([MLP(input_dim=d_model, hidden_dim=dim_feedforward, output_dim=d_model, num_layers=2) for i in range(self.n_shared_experts)])
    
    def forward(self, input, cond_route, emb_refine_text, selected_expert_id=-1):

        if self.use_refine_text:
            assert emb_refine_text is not None
        else:
            assert emb_refine_text is None

        if not self.batch_first:
            input = input.permute(1, 0, 2)
            cond_route = cond_route.permute(1, 0, 2)
            if emb_refine_text is not None:
                emb_refine_text = emb_refine_text.permute(1, 0, 2)  # [bs, 3, embed_dim]

        identity = input.clone()
        orig_shape = input.shape  # [bs, seqlen, embed_dim]

        topk_idx, topk_weight, aux_loss, scores, aux_loss_t, aux_loss_a = self.gate(cond_route) # cond_route: [bs, 1, topk], topk_idx: [bs * 1, topk], topk_weight: [bs * 1, topk]

        flat_topk_idx = topk_idx.view(-1)
        if self.training:   # use moe
            input = input.repeat_interleave(self.num_experts_per_tok * self.num_group, dim=0)
            if emb_refine_text is not None:
                emb_refine_text = emb_refine_text.repeat_interleave(self.num_experts_per_tok * self.num_group, dim=0)  # [bs * topk, 3, embed_dim]
            y = torch.empty_like(input, dtype=input.dtype)
            for i, expert in enumerate(self.experts): 
                if self.use_refine_text:
                    if (i//self.num_expert) == 0:
                        text_index = 0
                    elif (i//self.num_expert) == 1:
                        text_index = 1
                    else:
                        text_index = 2
                    # print(text_index)
                    y[flat_topk_idx == i] = expert(input[flat_topk_idx == i], emb_refine_text[flat_topk_idx == i][:, text_index, :]).float()
                else:
                    y[flat_topk_idx == i] = expert(input[flat_topk_idx == i]).float()

            y = (y.view(*topk_weight.shape, orig_shape[1], -1) * topk_weight.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
            self.aux_loss = aux_loss
            self.aux_loss_t = aux_loss_t
            self.aux_loss_a = aux_loss_a
        else:
            if selected_expert_id == -1:
                input = input.repeat_interleave(self.num_experts_per_tok * self.num_group, dim=0)
                if emb_refine_text is not None:
                    emb_refine_text = emb_refine_text.repeat_interleave(self.num_experts_per_tok * self.num_group, dim=0)  # [bs * topk, 3, embed_dim]
                y = torch.empty_like(input, dtype=input.dtype)
                for i, expert in enumerate(self.experts): 
                    if self.use_refine_text:
                        if (i//self.num_expert) == 0:
                            text_index = 0
                        elif (i//self.num_expert) == 1:
                            text_index = 1
                        else:
                            text_index = 2
                        y[flat_topk_idx == i] = expert(input[flat_topk_idx == i], emb_refine_text[flat_topk_idx == i][:, text_index, :]).float()  
                    else:
                        y[flat_topk_idx == i] = expert(input[flat_topk_idx == i]).float()
                y = (y.view(*topk_weight.shape, orig_shape[1], -1) * topk_weight.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                y =  y.view(*orig_shape)
            else:  # we don't use
                raise NotImplementedError("selected_expert_id is not supported in inference mode")
                if selected_expert_id//3 == 0:
                    text_index = 0
                elif selected_expert_id//3 == 1:
                    text_index = 1
                else:
                    text_index = 2
                y = self.experts[selected_expert_id](input, emb_refine_text[:, text_index, :]).float()  # [bs, seqlen, embed_dim]
            
        if self.n_shared_experts is not None:
            for i, layer in enumerate(self.shared_experts):
                # print("use shared expert: ", i)
                y = y + (1.0 / self.n_shared_experts) * layer(identity)

        if not self.batch_first:
            y = y.permute(1, 0, 2)
        
        if self.return_topk_idx:
            topk_idx = topk_idx.reshape(cond_route.shape[0], cond_route.shape[1], -1)  # [bs, seqlen, topk]
            topk_weight = topk_weight.reshape(cond_route.shape[0], cond_route.shape[1], -1)  # [bs, seqlen, topk]
            scores = scores.reshape(cond_route.shape[0], cond_route.shape[1], -1)  # [bs, seqlen, n_routed_experts]
            return y, topk_idx, topk_weight, scores
        
        return y


    def get_aux_loss(self):

        return self.aux_loss, self.aux_loss_t, self.aux_loss_a
    
    def get_metrics(self):
        if hasattr(self, 'metrics'):
            return self.metrics
        else:
            return None
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


if __name__ == "__main__":
    # debug
    seqTransEncoderLayer = TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_experts=3,
            num_shared_experts=None,
            num_experts_per_tok=1,
            dropout=0.1,
            activation="gelu",
            need_weights=True,
        )
    seqTransEncoder = TransformerEncoder(seqTransEncoderLayer, num_layers=1)

    print(seqTransEncoder)

    input = torch.randn(10, 3, 256)  # [seqlen, bs, embed_dim]
    cond_router = torch.randn(1, 3, 256*2)  # [seqlen, bs, embed_dim]
    emb_refine_text = torch.randn(3, 3, 256)  # [seq, bs, embed_dim]
    output  = seqTransEncoder(input, cond_router, emb_refine_text)

    print(output[0].shape)  # [seqlen, bs, embed_dim]
