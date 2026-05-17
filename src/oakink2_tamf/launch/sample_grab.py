import os
import numpy as np
import torch
import json
from copy import deepcopy
import pickle

import logging
import argparse
from config_reg import ConfigRegistry, ConfigEntrySource
from config_reg import (
    ConfigEntryCommandlineBoolPattern,
    ConfigEntryCommandlineSeqPattern,
)
from config_reg.callback import abspath_callback
from dev_fn.upkeep import log
from dev_fn.upkeep import ckpt
from dev_fn.upkeep.opt import argdict_to_string
from oakink2_tamf.util import log_suppress
from oakink2_tamf.launch.param import reg_mano_param, reg_model_param, reg_vae_model_param
from .param.model import reg_one_stage_model_param
from dev_fn.util.console_io import suppress_trimesh_logging

from oakink2_tamf.dataset.grab_dataset import GRAB
from dev_fn.transform.cast import map_copy_select_to
from ..model.pointnet_feature import PointNetfeat
from oakink2_tamf.model.MoE_model import MoEDiff, MoEDiff_multi_group
from oakink2_tamf.model.diffusion_util import create_gaussian_diffusion
from oakink2_tamf.dataset.collate import interaction_segment_collate


import torch.multiprocessing as mp

import manotorch
from manotorch.manolayer import ManoLayer
import time
import tqdm


PROG = "debug"

_logger = logging.getLogger(__name__)

PROG = PROG = os.path.splitext(os.path.basename(__file__))[0]
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__DEBUG = "debug"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    # override default
    config_reg.meta_info["exp_id"].default = "main"

    # base
    config_reg.register(
        "refine_text_path",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "data_path",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/data_single_hand.pkl",
    )
    config_reg.register(
        "data_obj_pc_path",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/obj.pkl",
    )
    config_reg.register(
        "text_json",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/text.json",
    )

    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # model related, weights
    reg_model_param(config_reg, "model")
    config_reg.register(
        "num_experts",
        prefix="model",
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=6,
    )
    config_reg.register(
        "num_shared_experts",
        prefix="model",
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=None,
    )
    config_reg.register(
        "num_experts_per_tok",
        prefix="model",
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=2,
    )
    config_reg.register(
        "return_topk_idx",
        prefix="model",
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        default=False,
    )
    config_reg.register(
        "use_group",
        prefix="model",
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        default=True,
    )


    config_reg.register(
        "model_weight_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )
    config_reg.register(
        "sample_save_offset",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
    )
    config_reg.register(
        "ddim_step",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=None,
    )
    config_reg.register(
        "num_worker",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=3,
    )
    config_reg.register(
        "device_id",
        prefix=PARAM_PREFIX__RUNTIME,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[0, 1, 2, 3],
    )



def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [
        PARAM_PREFIX__DATA,
        PARAM_PREFIX__DEBUG,
        "mano",
        "model",
        PARAM_PREFIX__RUNTIME,
    ]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res

def proc_obj_feat_final_train(obj_scale, obj_cent, obj_feat, use_obj_scale_centroid):
    obj_feat_global = obj_feat[:, 0, :1024]  # (B, 1024)
    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_global, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_global
    return obj_feat_final


def sample_worker(
    log_queue,
    worker_id,
    num_worker,
    device_id,
    ckpt_cfg,
    run_cfg,
):
    log.configure_mp_worker(log_queue, worker_id)
    _logger.info("worker_id: %02d", worker_id)
    _logger.info("device_id: %d", device_id)

    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    # dataset
    if run_cfg["data"]["refine_text_path"] is not None:
        use_refine_text = True
        refine_text_path = os.path.join(run_cfg["data"]["refine_text_path"], "test_data_refine.json")
        _logger.info("refine_text_path: %s", refine_text_path)
    else:
        use_refine_text = False
        refine_text_path = None
        _logger.info("Don't use refine_text_path")

    all_dataset = GRAB(
        data_path=run_cfg["data"]["data_path"], 
        data_obj_pc_path=run_cfg["data"]["data_obj_pc_path"],
        text_json=run_cfg["data"]["text_json"],
        refine_text_path=refine_text_path,
    )

    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float32

    
    # model
    model_cfg = run_cfg["model"]
    model = MoEDiff_multi_group(
        input_dim=model_cfg["input_dim"],
        obj_input_dim=model_cfg["obj_input_dim"],
        hand_shape_dim=model_cfg["hand_shape_dim"],
        obj_embed_dim=model_cfg["obj_embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        ff_size=model_cfg["ff_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        num_experts=run_cfg["model"]["num_experts"],
        num_shared_experts=run_cfg["model"]["num_shared_experts"],
        num_experts_per_tok=run_cfg["model"]["num_experts_per_tok"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        use_refine_text=use_refine_text,
        use_group=model_cfg["use_group"],
        return_topk_idx=run_cfg["model"]["return_topk_idx"],
    ).to(device)
    
    if run_cfg["debug"]["ddim_step"] is not None:
        _logger.info("Using ddim_step: %s", run_cfg["debug"]["ddim_step"])
        use_ddim = True
        diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine", timestep_respacing=run_cfg["debug"]["ddim_step"])
    else:
        use_ddim = False
        diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine")
    state_dict = torch.load(run_cfg["debug"]["model_weight_filepath"], map_location=device)
    missing_key_list, unexpected_key_list = model.load_state_dict(state_dict, strict=False)
    missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
    missing_key_list = [k for k in missing_key_list if not k.startswith("long_clip_model")]

    # pointfeat
    _logger.info("build point encoder")
    _point_encoder = PointNetfeat(global_feat=False, feature_transform=False, in_dim=3).to(device)
    weight_path = "/data/data0/xuhang/Text2HOI/checkpoints/grab/pointfeat.pth"
    checkpoints = torch.load(weight_path, map_location=device)
    _point_encoder.load_state_dict(checkpoints["model"])
    _point_encoder.eval()
    for p in _point_encoder.parameters():
        p.requires_grad = False


    if worker_id == 0:
        _logger.info("missing_keys: %s", missing_key_list)
        _logger.info("unexpected_keys: %s", unexpected_key_list)

    worker_sample_id_start = int(len(all_dataset) * worker_id / num_worker)
    worker_sample_id_stop = int(len(all_dataset) * (worker_id + 1) / num_worker)
    _logger.info("%06d %06d", worker_sample_id_start, worker_sample_id_stop)

    saved_path = os.path.join(ckpt_path, "sample", run_cfg["debug"]["sample_save_offset"])
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    saved_list = []
    for file in os.listdir(saved_path):
        saved_list.append(int(file.split('.')[0]))
    print(len(saved_list))

    duplicate_check = set()
    for sample_id in tqdm.tqdm(range(worker_sample_id_start, worker_sample_id_stop), ncols=150):
        if sample_id in saved_list:
            continue
        gt_sample = all_dataset[sample_id]
        info = gt_sample["info"]
        if info in duplicate_check:
            print(f"Duplicate sample found: {info}, skipping...")
            continue
        duplicate_check.add(info)

        # debug
        gt_batch = interaction_segment_collate([gt_sample])
        batch_device = map_copy_select_to(
            gt_batch,
            device=device,
            dtype=dtype,
            select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "normalized_obj_pc", "obj_cent", "obj_scale"),
        )
        # process obj feature
        with torch.no_grad():
            obj_feat = _point_encoder(batch_device["normalized_obj_pc"])
        obj_feat_final = proc_obj_feat_final_train(batch_device["obj_scale"], batch_device["obj_cent"], obj_feat, use_obj_scale_centroid=True)  # (B, 1028)
        # print(obj_feat_final.shape)
        batch_device["obj_embedding"] = obj_feat_final.unsqueeze(1)  # (B, 1, 1028)  # to align with oakink2


        start_time = time.time()
        with torch.no_grad():
            sample_fn = diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            model.eval()
            input_shape = tuple(batch_device["pose_repr"].shape)
            input_shape = (input_shape[0], input_shape[2], 1, input_shape[1])
            sample = sample_fn(
                model,
                input_shape,  # adapt from mdm
                clip_denoised=False,
                model_kwargs={"batch": batch_device},
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        sample_ = sample.permute((0, 3, 1, 2)) 
        sample_np = sample_.detach().clone().cpu().numpy()
        pose_repr_sample_np = sample_np.squeeze(3).squeeze(0)  # (160, 99)
        end_time = time.time()
        # _logger.info("sample time: %.3f", end_time - start_time)


        _info = info.split('_')[0]+'_'+info.split('_')[1]
        action_name = gt_sample['primitive']
        hand_side = gt_sample["hand_side"]
        obj_traj = gt_sample["obj_traj"][0]
        # print(obj_traj.shape)

        save_dict = {
            "info": (_info, action_name, hand_side),
            "hand_side": hand_side,
            "gt_hand": gt_sample["pose_repr"],
            "pred_hand": pose_repr_sample_np,
            "mask_hand": gt_sample["mask"],
            "shape": gt_sample["shape"],
            "text": gt_sample["text"],
            "obj_traj": obj_traj,
            "obj_name": gt_sample["obj_list"][0],
            "action_name": action_name,
        }

        if commit:
            save_dict_filepath = os.path.join(
                ckpt_path, "sample_grab", run_cfg["debug"]["sample_save_offset"],
                f"{_info}_{action_name}",
                f"{hand_side}",
                "save_dict.pkl",
            )
            os.makedirs(os.path.dirname(save_dict_filepath), exist_ok=True)
            with open(save_dict_filepath, "wb") as ofstream:
                pickle.dump(save_dict, ofstream)
        
        _logger.info("sample %06d", sample_id)


def main():
    config_reg = ConfigRegistry(prog=PROG)
    ckpt.reg_entry(config_reg)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    ckpt_cfg = ckpt.reg_extract(config_reg)
    run_cfg = reg_extract(config_reg)

    ckpt.ckpt_setup(ckpt_cfg)
    ckpt.ckpt_opt(ckpt_cfg, ckpt=ckpt_cfg, run=run_cfg)

    log.log_init()
    log.enable_console()
    _logger.info("ckpt_cfg: %s", argdict_to_string(ckpt_cfg))
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    num_worker = run_cfg["runtime"]["num_worker"]
    device_id_list = run_cfg["runtime"]["device_id"]

    # set multiprocessing context
    mp.set_start_method("spawn")
    log_queue = mp.Queue()
    log.configure_mp_main(log_queue)

    process_list = []
    for worker_id in range(num_worker):
        worker_device_id = device_id_list[worker_id % len(device_id_list)]
        process_list.append(
            mp.Process(
                target=sample_worker,
                kwargs=dict(
                    log_queue=log_queue,
                    worker_id=worker_id,
                    num_worker=num_worker,
                    device_id=worker_device_id,
                    ckpt_cfg=ckpt_cfg,
                    run_cfg=run_cfg,
                ),
            )
        )
    for worker_id in range(num_worker):
        process_list[worker_id].start()

    for worker_id in range(num_worker):
        process_list[worker_id].join()

    log.deconfigure_mp_main(log_queue)
    _logger.info("conclude parallel worker")


if __name__ == "__main__":
    main()
