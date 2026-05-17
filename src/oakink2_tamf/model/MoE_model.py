import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from thirdparty.LongCLIP.model import longclip
import os
from oakink2_tamf.model.MoEtransformer_multi_group import TransformerEncoderLayer, TransformerEncoder


_logger = logging.getLogger(__name__)


class MoEDiff_multi_group(nn.Module):    # for text group
    def __init__(
        self,
        input_dim=99,
        obj_input_dim=9,
        hand_shape_dim=10,
        obj_embed_dim=768,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        num_experts=6,
        num_shared_experts=None,
        num_experts_per_tok=2,
        dropout=0.1,
        activation="gelu",
        use_refine_text=True,
        use_group=True,
        clip_dim=512,
        long_clip_dim=512,
        clip_version="ViT-B/32",
        long_clip_version="longclip-B",
        return_topk_idx=False,
        **kargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.long_clip_dim = long_clip_dim

        self.input_feats = input_dim
        self.obj_input_feats = obj_input_dim

        self.hand_shape_feats = hand_shape_dim
        self.obj_embed_feats = obj_embed_dim

        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.hand_side_process = HandsideProcess(self.latent_dim)
        self.hand_shape_process = HandShapeProcess(self.hand_shape_feats, self.latent_dim)
        self.obj_embed_process = ObjectEmbedProcess(self.obj_embed_feats, self.latent_dim)

        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.obj_input_process = ObjectInputProcess(self.obj_input_feats, self.latent_dim)
        self.input_merge = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        _logger.info("TRANS_ENC init")
        self.return_topk_idx = return_topk_idx
        _logger.info("return_topk_idx: %s", return_topk_idx)

        if use_group:
            seqTransEncoderLayer = TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                dropout=self.dropout,
                activation=self.activation,
                return_topk_idx=self.return_topk_idx, # return_topk_idx=False during training for MoE, True during inference
                use_refine_text=use_refine_text,  
            )
            self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        _logger.info("EMBED TEXT")
        _logger.info("Loading CLIP...")
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

        self.use_refine_text = use_refine_text
        # long clip
        if use_refine_text:
            assert use_group==True, "refine text is only used when use_group is True"
            _logger.info("Loading LONG CLIP...")
            self.long_clip_model = self.load_and_freeze_long_clip(long_clip_version)
            self.embed_hand_text = nn.Linear(self.long_clip_dim, self.latent_dim)
            self.embed_finger_text = nn.Linear(self.long_clip_dim, self.latent_dim)
            self.embed_joint_text = nn.Linear(self.long_clip_dim, self.latent_dim)

        self.output_process = OutputProcess(self.input_feats, self.latent_dim)
        self.aux_loss_list = []
        self.aux_loss_list_t = []
        self.aux_loss_list_a = []

        # action
        self.embed_action = nn.Linear(self.clip_dim, self.latent_dim)

        if self.return_topk_idx:
            self.topk_idx_list = []
            self.topk_weight_list = []
            self.score_list = []


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def load_and_freeze_long_clip(self, long_clip_version):
        long_clip_checkpoint = 'thirdparty/LongCLIP/checkpoints/' + long_clip_version + '.pt'
        long_clip_model, clip_preprocess = longclip.load(
            long_clip_checkpoint, device="cpu"
        )  

        # Freeze CLIP weights
        long_clip_model.eval()
        for p in long_clip_model.parameters():
            p.requires_grad = False

        return long_clip_model


    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device
            )
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def encode_long_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        texts = longclip.tokenize(raw_text, truncate=True).to(
            device
        )  # [bs, context_length] 
        return self.long_clip_model.encode_text(texts).float()


    def forward(self, x, timesteps, batch):
        """
        x: [TODO]
        timesteps: [batch_size] (int) for diffusion
        """
        batch_size = x.shape[0]
        input = x.clone()

        emb_list = []
        emb_timestep = self.embed_timestep(timesteps)  # [1, bs, d]
        emb_list.append(emb_timestep)

        enc_text = self.encode_text(batch["text"])
        emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=False))  # [bs, d]
        emb_text = emb_text.reshape((1, batch_size, self.latent_dim))
        emb_list.append(emb_text)
        # print(timesteps)

        # long_text
        if self.use_refine_text:
            hand_level_text = self.encode_long_text(batch["hand_level_text"])
            finger_level_text = self.encode_long_text(batch["finger_level_text"])
            joint_level_text = self.encode_long_text(batch["joint_level_text"])
            emb_hand_text = self.embed_hand_text(self.mask_cond(hand_level_text, force_mask=False))  # [bs, d]
            emb_hand_text = emb_hand_text.reshape((1, batch_size, self.latent_dim))
            emb_finger_text = self.embed_finger_text(self.mask_cond(finger_level_text, force_mask=False))  # [bs, d]
            emb_finger_text = emb_finger_text.reshape((1, batch_size, self.latent_dim))
            emb_joint_text = self.embed_joint_text(self.mask_cond(joint_level_text, force_mask=False))  # [bs, d]
            emb_joint_text = emb_joint_text.reshape((1, batch_size, self.latent_dim))
            emb_refine_text = torch.cat(
                (emb_hand_text, emb_finger_text, emb_joint_text), dim=0
            ) # [3, bs, d]
        else:
            emb_refine_text = None

        # action
        # print("batch['primitive']", batch["primitive"])
        enc_action = self.encode_text(batch["primitive"])
        emb_action = self.embed_action(self.mask_cond(enc_action, force_mask=False))  # [bs, d]
        emb_action = emb_action.reshape((1, batch_size, self.latent_dim))
        emb_list.append(emb_action)

        emb_handside = self.hand_side_process(batch["hand_side"])  # [1, bs, d]
        emb_list.append(emb_handside)
        emb_shape = self.hand_shape_process(batch["shape"])  # [1, bs, d]
        emb_list.append(emb_shape)

        emb_obj = self.obj_embed_process(batch["obj_embedding"])  # [1, bs, d]
        emb_list.append(emb_obj)
        emb = torch.cat(emb_list, dim=0)  # [5, bs, d]
        emb = torch.nan_to_num(emb)
        emb_prefix_len = emb.shape[0]

        hand_traj = self.input_process(x)  # [seq_len, bs, d]
        object_input = self.obj_input_process(batch["obj_traj"])  # [seq_len, bs, d]
        # brute force merger
        merged_input = torch.cat((hand_traj, object_input), dim=-1)
        x = self.input_merge(merged_input)  # [seq_len, bs, d]
        x = torch.nan_to_num(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+5, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+5, bs, d]

        # use timestep embedding as the condition
        # cond_router = emb_timestep
        # use timestep and action embedding as the condition
        cond_router = torch.cat((emb_timestep, emb_action), dim=-1)  # [1, bs, d*2]

        if self.return_topk_idx:
            # during inference, we return the topk indices
            # this is used to select the experts for each token
            output, aux_loss_list_tuple, _, all_topk_idx, all_topk_weight, all_scores = self.seqTransEncoder(xseq, cond_router, emb_refine_text)
            output = output[emb_prefix_len:]  # [seqlen, bs, d]
            # save the topk indices and weights
            all_topk_idx_np = all_topk_idx[:, 0, :, :].detach().cpu().numpy()  # [8, 1, topk]
            all_topk_weight_np = all_topk_weight[:, 0, :, :].detach().cpu().numpy()
            all_scores_np = all_scores[:, 0, :, :].detach().cpu().numpy()  # [8, 1, num_experts]
            self.topk_idx_list.append(all_topk_idx_np)
            self.topk_weight_list.append(all_topk_weight_np)
            self.score_list.append(all_scores_np)
            if timesteps[0] == 0:
                # assert len(self.topk_idx_list) == len(self.topk_weight_list) == len(self.score_list) == 1000
                save_dict = {
                    "text": batch["text"][0],
                    "info": batch["info"][0],
                    "primitive": batch["primitive"][0],
                    "topk_idx_inverse": np.array(self.topk_idx_list),
                    "topk_weight_inverse": np.array(self.topk_weight_list),
                    "score_inverse": np.array(self.score_list),
                }
                # name = "arctic__final_change_gate_with_smooth_v1"
                name = "h2o__final"
                if name[:6] == "arctic":
                    _info = batch["info"][0]
                    info = (_info.split("_")[0]+"_"+_info.split("_")[1], _info.split("_")[0]+"_"+_info.split("_")[1], _info.split("_")[2])
                    save_dict_filepath = os.path.join(
                            "/data/data4/xuhang/OakInk2-TaMF/moe_output", "save_all_step", name, "0099_ddim100_viz_moe",
                            str(info[0]).replace('/', '++'),
                            str(info[1]),
                            str(info[2]),
                            "output.pkl"
                        )
                else:
                    _info = batch["info"][0]
                    info = (_info.split("_")[0]+"_"+_info.split("_")[1], _info.split("_")[0]+"_"+_info.split("_")[1], _info.split("_")[2])
                    save_dict_filepath = os.path.join(
                            "/data/data4/xuhang/OakInk2-TaMF/moe_output", "save_all_step", name, "0399_ddim100_viz_moe",
                            str(info[0]).replace('/', '++'),
                            str(info[1]),
                            str(info[2]),
                            "output.pkl"
                        )
                os.makedirs(os.path.dirname(save_dict_filepath), exist_ok=True)
                with open(save_dict_filepath, "wb") as f:
                    import pickle
                    pickle.dump(save_dict, f)
                self.topk_idx_list = []
                self.topk_weight_list = []
                self.score_list = []

        else:
            output, aux_loss_list_tuple, _ = self.seqTransEncoder(xseq, cond_router, emb_refine_text)  # [seqlen, bs, d]
            output = output[emb_prefix_len:]

        output = self.output_process(output)  # [bs, input_dim, 1, nframes]
        output = torch.nan_to_num(output)

        if self.training:
            aux_loss_list = aux_loss_list_tuple[0]
            aux_loss_list_t =  aux_loss_list_tuple[1]
            aux_loss_list_a = aux_loss_list_tuple[2]
            self.aux_loss_list = [aux_loss.item() for aux_loss in aux_loss_list]
            self.aux_loss_list_t = [aux_loss_t.item() for aux_loss_t in aux_loss_list_t]
            self.aux_loss_list_a = [aux_loss_a.item() for aux_loss_a in aux_loss_list_a]
        else:
            aux_loss_list = aux_loss_list_tuple[0]
            aux_loss_list_t =  aux_loss_list_tuple[1]
            aux_loss_list_a = aux_loss_list_tuple[2]
            self.aux_loss_list = [aux_loss for aux_loss in aux_loss_list]
            self.aux_loss_list_t = [aux_loss_t for aux_loss_t in aux_loss_list_t]
            self.aux_loss_list_a = [aux_loss_a for aux_loss_a in aux_loss_list_a]

        return output

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.clip_model.eval()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats, _, nframes = x.shape  # [bs, in_dim, 1, seqlen]
        x = x.permute((3, 0, 1, 2))  # [seqlen, bs, i, 1]
        x = x.reshape((nframes, bs, nfeats))  # [seqlen, bs, i]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class ObjectInputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.hidden_dim = hidden_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)  # TODO: use a MLP

    def forward(self, x):
        bs, nobj, seqlen, nfeat = x.shape
        x = x.permute((0, 2, 1, 3))  # [bs, seqlen, nobj, inp]
        x = self.poseEmbedding(x)  # [bs, seqlen, nobj, d]
        # avg pool
        x = torch.mean(x, dim=2)  # [bs, seqlen, d]
        x = x.permute((1, 0, 2))  # [seqlen, bs, d]
        return x


class ObjectEmbedProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nobj, dim_obj_feat = x.shape
        x = torch.mean(x, dim=1)  # average -> [bs, dim_obj_feat]
        x = self.embedding(x)  # [bs, d]
        x = x.unsqueeze(0)  # [1, bs, d]
        return x


class HandsideProcess(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        _rh = torch.zeros((self.latent_dim,), dtype=torch.float32)
        self.register_buffer("rh_embed", _rh)
        _lh = torch.zeros((self.latent_dim,), dtype=torch.float32)
        _lh[0] = 1.0
        self.register_buffer("lh_embed", _lh)

    def forward(self, hand_side):
        res = []
        for hs in hand_side:
            if hs == "rh":
                embed = self.rh_embed
            elif hs == "lh":
                embed = self.lh_embed
            else:
                raise ValueError(f"unexpected hand_side: {hs}")
            res.append(embed)
        res = torch.stack(res, dim=0)  # [bs, d]
        res = res.unsqueeze(0)  # [1, bs, d]
        return res


class HandShapeProcess(nn.Module):
    def __init__(self, shape_dim, latent_dim) -> None:
        super().__init__()
        self.shape_dim = shape_dim
        self.latent_dim = latent_dim
        self.shape_embed = nn.Linear(self.shape_dim, self.latent_dim)

    def forward(self, shape):
        # shape [B, SEQLEN, 10]
        shape_avg = torch.mean(shape, dim=1)  # [bs, 10]
        res = self.shape_embed(shape_avg)  # [bs, d]
        res = res.unsqueeze(0)  # [1, bs, d]
        return res


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, input_dim]
        output = output.reshape(nframes, bs, self.input_feats, 1)
        output = output.permute(1, 2, 3, 0)  # [bs, inp, 1, nframes]
        return output
    

    
# 统计指标，总参数量，激活参数量

def count_parameters(model: nn.Module):
    """统计模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }

import re

def find_and_analyze_moe_layers(
    model: nn.Module, 
    expert_container_name: str
):
    """
    自动发现并分析模型中所有 MoE 层的参数分布。

    Args:
        model (nn.Module): 您的 PyTorch 模型。
        expert_container_name (str): 用于唯一识别专家列表模块的名称片段。
                                     例如: "FFN.experts"。

    Returns:
        Dict[str, Any]: 一个包含详细分析结果的字典，包括总体统计和每层详情。
    """
    moe_layers_info = []
    # 1. 发现阶段：遍历所有模块，找到所有 MoE 专家容器
    for name, module in model.named_modules():
        # 我们寻找名字以 expert_container_name 结尾的 ModuleList
        if name.endswith(f".{expert_container_name}") and isinstance(module, nn.ModuleList):
            moe_layers_info.append({
                "path": name,
                "num_experts": len(module)
            })
    
    if not moe_layers_info:
        raise ValueError(f"在模型中未能找到任何名为 '{expert_container_name}' 的专家容器。请检查名称。")

    # 2. 参数分类与统计阶段
    shared_params = 0
    total_expert_params = 0
    
    # 初始化每层的专家参数列表
    per_layer_stats = {
        info['path']: {
            "num_experts": info['num_experts'],
            "expert_params": [0] * info['num_experts'],
            "total_expert_params": 0
        } for info in moe_layers_info
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_expert_param = False
        # 检查该参数是否属于已发现的任何一个 MoE 层
        for info in moe_layers_info:
            moe_path = info['path']
            if name.startswith(moe_path):
                is_expert_param = True
                
                # 提取专家索引
                # pattern looks for ".<expert_idx>." immediately after the moe_path
                pattern = re.compile(f"^{re.escape(moe_path)}\\.(\\d+)\\.")
                match = pattern.search(name)
                if match:
                    expert_idx = int(match.group(1))
                    per_layer_stats[moe_path]['expert_params'][expert_idx] += param.numel()
                break # 参数归属已确定，跳出内层循环
        
        if not is_expert_param:
            shared_params += param.numel()

    # 3. 聚合与格式化结果
    for path, stats in per_layer_stats.items():
        layer_total_expert_params = sum(stats['expert_params'])
        stats['total_expert_params'] = layer_total_expert_params
        total_expert_params += layer_total_expert_params
        
    total_params = shared_params + total_expert_params

    return {
        "overall_stats": {
            "total_params": total_params,
            "shared_params": shared_params,
            "total_expert_params": total_expert_params,
            "num_moe_layers": len(moe_layers_info)
        },
        "per_layer_stats": per_layer_stats
    }


def calculate_weighted_average_params(
    all_scores_list: list,
    shared_params: int,
    expert_params_list: list
) -> dict:
    """
    根据一个数据集上所有样本的激活分数，计算加权的平均激活参数量。

    Args:
        all_scores_list (list): 包含多个批次 scores 张量的列表。
                                每个张量形状为 [N, Time, Total_Experts]。
        shared_params (int): 模型的共享参数量。
        expert_params_list (list): 包含每个专家参数量的列表。
    """
    if not all_scores_list:
        return {}

    # 1. 将所有批次的 scores 张量拼接并展平
    all_scores_tensor = torch.cat([torch.tensor(s) for s in all_scores_list], dim=0)
    # 展平为 [Total_Tokens, Total_Experts]
    flat_scores = all_scores_tensor.view(-1, all_scores_tensor.shape[-1])
    
    # 2. 将专家参数列表转换为张量
    expert_params_tensor = torch.tensor(expert_params_list, 
                                        device=flat_scores.device, 
                                        dtype=torch.float32)

    # 3. 计算每个 token 的加权激活专家参数量
    #    这是核心步骤：(N, E) @ (E, 1) -> (N, 1)
    #    使用 element-wise multiplication and sum 效率更高
    weighted_expert_params_per_token = (flat_scores * expert_params_tensor).sum(dim=1)
    
    # 4. 计算整个数据集上的平均值
    avg_weighted_expert_params = weighted_expert_params_per_token.mean().item()
    
    # 5. 计算总的平均激活参数量
    total_weighted_active_params = shared_params + avg_weighted_expert_params

    return {
        "total_weighted_active_params": total_weighted_active_params,
        "avg_weighted_expert_params": avg_weighted_expert_params,
        "total_tokens_analyzed": flat_scores.shape[0]
    }


if __name__ == "__main__":
    # 测试代码
    model = MoEDiff_multi_group(
        input_dim=99,
        obj_input_dim=9,
        hand_shape_dim=10,
        obj_embed_dim=1028,
        latent_dim=512,
        ff_size=2048,
        num_layers=8,
        num_heads=4,
        num_experts=3,
        num_shared_experts=None,
        num_experts_per_tok=1,
        dropout=0.1,
        activation="gelu",
        use_refine_text=True,
        use_group=True,
        clip_dim=512,
        long_clip_dim=512,
        clip_version="ViT-B/32",
        long_clip_version="longclip-B",
        return_topk_idx=False,
    )
    # print(model)

    params_info = count_parameters(model)
    print("-" * 50)
    print("模型参数量 (Model Parameter Counts):")
    print(f"   - 总参数: {params_info['total_params'] / 1e6:.2f} M")
    print(f"   - 可训练参数: {params_info['trainable_params'] / 1e6:.2f} M")


    moe_stats = find_and_analyze_moe_layers(
            model=model,
            expert_container_name="FFN.experts"
        )
    # --- 打印结果 ---
    print("-" * 50)
    print("多层 MoE 模型参数分析:")
    overall = moe_stats['overall_stats']
    per_layer = moe_stats['per_layer_stats']

    print("\n" + "="*50)
    print("总体统计 (Overall Stats)")
    print("-" * 50)
    print(f"发现的 MoE 层数量: {overall['num_moe_layers']}")
    print(f"模型总参数量: {overall['total_params'] / 1e6:.2f} M")
    print(f"  - 共享参数总量: {overall['shared_params'] / 1e6:.2f} M")
    print(f"  - 所有专家参数总量: {overall['total_expert_params'] / 1e6:.2f} M")
    print("="*50)

    print("\n各 MoE 层详细统计 (Per-Layer Stats)")
    print("-" * 50)
    for path, stats in per_layer.items():
        print(f"MoE 层路径: {path}")
        print(f"  - 专家数量: {stats['num_experts']}")
        print(f"  - 该层专家参数总量: {stats['total_expert_params'] / 1e6:.2f} M")
        # 如果需要，可以取消注释以查看每个专家的详细信息
        for i, p_count in enumerate(stats['expert_params']):
            print(f"    - 专家 {i}: {p_count / 1e3:.2f} K")
        print("-" * 20)



    # # 假设我们有一些模拟的 all_scores 数据
    # simulated_all_scores = [
    #     np.random.rand(8, 100, 6),  # 模拟一个批次的 scores
    #     np.random.rand(8, 100, 6)   # 再模拟一个批次的 scores
    # ]

    # weighted_avg_params = calculate_weighted_average_params(
    #     simulated_all_scores,
    #     shared_params=moe_param_stats["shared_params"],
    #     expert_params_list=moe_param_stats["expert_params"]
    # )
    # print("加权平均激活参数量统计:", weighted_avg_params)