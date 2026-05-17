import os
import numpy as np
import torch
import cv2
import json

import logging
import argparse
import itertools
import shutil
import tqdm
import time
import pickle

import torch.multiprocessing as mp
import torch.distributed as dist
from ..util import ddp_util

from config_reg import ConfigRegistry, ConfigEntrySource
from config_reg import (
    ConfigEntryCommandlineBoolPattern,
    ConfigEntryCommandlineSeqPattern,
)
from config_reg.callback import abspath_callback
from dev_fn.upkeep import log
from ..util import log_suppress
from dev_fn.upkeep import ckpt
from dev_fn.upkeep.opt import argdict_to_string
from dev_fn.upkeep.config import cb__decode_file, cls__cb__link_bool_opt

from dev_fn.util.console_io import suppress_trimesh_logging
from dev_fn.util import pbar_util
from dev_fn.util import random_util

from dev_fn.transform.cast import map_copy_select_to
from oakink2_tamf.dataset.grab_dataset import GRAB
from ..dataset.collate import interaction_segment_collate
from ..model.MoE_model import MoEDiff_multi_group
from ..model.pointnet_feature import PointNetfeat
from ..model.loss_grab import GenModelLoss
from ..model.diffusion_util import create_gaussian_diffusion
from ..model.diffusion.resample import create_named_schedule_sampler
from ..util.net_util import clip_gradient
from ..util.state_util import save_state
from ..util.summary_writer import DDPSummaryWriter

from .param import reg_mano_param, reg_loss_param, reg_model_param
from .param.loss_refine import our_reg_loss_param
from dev_fn.transform.rotation import rot6d_to_rotmat, rotmat_to_quat
from pytorch3d.structures import Meshes
from manotorch.manolayer import ManoLayer

_logger = logging.getLogger(__name__)

PROG = "train_grab"
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__TRAIN = "train"
PARAM_PREFIX__TEST = "test"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    config_reg.register(
        "refine_text_path",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # train
    config_reg.register(
        "data_path",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/train/data_single_hand.pkl",
    )
    config_reg.register(
        "data_obj_pc_path",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/train/obj.pkl",
    )
    config_reg.register(
        "text_json",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/train/text.json",
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=16,
    )
    config_reg.register(
        "num_epoch",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=400,
    )
    config_reg.register(
        "record_freq",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=5,
    )
    config_reg.register(
        "scheduler_milestone",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[150, 250],
    )
    config_reg.register(
        "scheduler_gamma",
        prefix=PARAM_PREFIX__TRAIN,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.5,
    )
    our_reg_loss_param(config_reg, f"{PARAM_PREFIX__TRAIN}.loss", WS_DIR)
    config_reg.register(
        "reload_ckpt_model_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "reload_ckpt_optimizer_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    # test
    config_reg.register(
        "data_path",
        prefix=PARAM_PREFIX__TEST,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/data_single_hand.pkl",
    )
    config_reg.register(
        "data_obj_pc_path",
        prefix=PARAM_PREFIX__TEST,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/obj.pkl",
    )
    config_reg.register(
        "text_json",
        prefix=PARAM_PREFIX__TEST,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="data_grab/grab/test/text.json",
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "test_freq",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=100,
    )

    # model
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

    ## runtime config
    config_reg.register(
        "num_worker",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_ONLY,
        default=2,
    )
    config_reg.register(
        "device_id",
        prefix=PARAM_PREFIX__RUNTIME,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[0],
    )
    config_reg.register(
        "seed",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [
        PARAM_PREFIX__DATA,
        PARAM_PREFIX__TRAIN,
        PARAM_PREFIX__TEST,
        PARAM_PREFIX__RUNTIME,
        "model",
        "mano",
    ]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def summary_extra_loss(summary_writer, loss_store, extra_loss_store, lr=None, step_id=None):
    time_curr = time.time()
    summary_writer.add_scalar(
        tag="diffusion_loss", value=float(loss_store["diffusion_loss"]), global_step=step_id, walltime=time_curr
    )
    for loss_name, loss_node in extra_loss_store.items():
        if loss_name == "loss":
            summary_writer.add_scalar(tag="extra_loss", value=float(loss_node), global_step=step_id, walltime=time_curr)
        else:
            summary_writer.add_scalar(
                tag=f"extra_loss/{loss_name}", value=float(loss_node), global_step=step_id, walltime=time_curr
            )
    if lr is not None:
        summary_writer.add_scalar(tag="lr", value=float(lr), global_step=step_id, walltime=time_curr)


def proc_obj_feat_final_train(obj_scale, obj_cent, obj_feat, use_obj_scale_centroid):
    obj_feat_global = obj_feat[:, 0, :1024]  # (B, 1024)
    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_global, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_global
    return obj_feat_final


def run(rank, world_size, ckpt_cfg, run_cfg):
    log.log_init()
    log.enable_console()
    ddp_util.log_status_update(rank)  # disable root_logger on non_zero rank

    ckpt.ckpt_setup(ckpt_cfg, rank=rank)
    ckpt.ckpt_opt(ckpt_cfg, rank=rank, world_size=world_size, ckpt=ckpt_cfg, run=run_cfg)

    _logger.info("world_size: %d", world_size)
    _logger.info("ckpt_cfg: %s", argdict_to_string(ckpt_cfg))
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    # device
    device_id_list = run_cfg["runtime"]["device_id"]
    device_id = device_id_list[rank] if rank is not None else device_id_list[0]
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    dtype = torch.float32
    ddp_util.setup_ddp(rank, world_size)

    # ckpt
    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    # dataset
    if run_cfg["data"]["refine_text_path"] is not None:
        use_refine_text = True
        train_refine_text_path = os.path.join(run_cfg["data"]["refine_text_path"], "train_data_refine.json")
        # test_refine_text_path = os.path.join(run_cfg["data"]["refine_text_path"], "test_data_refine.json")
        _logger.info("refine_text_path: %s", train_refine_text_path)
    else:
        use_refine_text = False
        train_refine_text_path = None
        test_refine_text_path = None
        _logger.info("Don't use refine_text_path")

    train_dataset = GRAB(
        data_path=run_cfg["train"]["data_path"], 
        data_obj_pc_path=run_cfg["train"]["data_obj_pc_path"], 
        text_json=run_cfg["train"]["text_json"],
        refine_text_path=train_refine_text_path,
    )
    if not rank:
        test_dataset = GRAB(
            data_path=run_cfg["test"]["data_path"], 
            data_obj_pc_path=run_cfg["test"]["data_obj_pc_path"],
            text_json=run_cfg["test"]["text_json"],
            refine_text_path=None,
        )
    else:
        test_dataset = None

    # handle batch_size
    world_batch_size = run_cfg["train"]["batch_size"]
    if world_batch_size % world_size != 0:
        _logger.warning("batch_size %d is not divisible by world_size %d", world_batch_size, world_size)
    batch_size = world_batch_size // world_size
    _logger.info("batch_size: %d | equiv world_batch_size %d", batch_size, batch_size * world_size)

    # handle data_loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=interaction_segment_collate,
        batch_size=batch_size,
        num_workers=run_cfg["runtime"]["num_worker"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=None,
    )
    if not rank:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=interaction_segment_collate,
            batch_size=run_cfg["test"]["batch_size"],
            num_workers=run_cfg["runtime"]["num_worker"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=None,
        )
    else:
        test_dataloader = None

    # model
    model_cfg = run_cfg["model"]
    model_ = MoEDiff_multi_group(
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
        use_group=run_cfg["model"]["use_group"],
        return_topk_idx=run_cfg["model"]["return_topk_idx"],
    ).to(device)
    # resume model
    now_epoch = 0
    if (reload_ckpt_model_filepath := run_cfg["train"]["reload_ckpt_model_filepath"]) is not None:
        now_epoch = int(reload_ckpt_model_filepath.split('/')[-1].split('.')[0].split('_')[1]) + 1
        _logger.info("model reload ckpt: %s", reload_ckpt_model_filepath)
        reload_ckpt_model_weight = torch.load(reload_ckpt_model_filepath, map_location=device)
        missing_key_list, unexpected_key_list = model_.load_state_dict(reload_ckpt_model_weight, strict=False)
        missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
        missing_key_list = [k for k in missing_key_list if not k.startswith("long_clip_model")]
        _logger.info("model reload ckpt missing_key_list: %s", missing_key_list)
        _logger.info("model reload ckpt unexpected_key_list: %s", unexpected_key_list)
    model = torch.nn.parallel.DistributedDataParallel(
        model_,
        device_ids=[device],
        output_device=device,
        # find_unused_parameters=True,
    )
    
    # pointfeat
    _logger.info("build point encoder")
    _point_encoder = PointNetfeat(global_feat=False, feature_transform=False, in_dim=3).to(device)
    weight_path = "/data/data0/xuhang/Text2HOI/checkpoints/grab/pointfeat.pth"
    checkpoints = torch.load(weight_path, map_location=device)
    _point_encoder.load_state_dict(checkpoints["model"])
    _point_encoder.eval()
    for p in _point_encoder.parameters():
        p.requires_grad = False
    
    model_extra_loss = GenModelLoss(
        run_cfg["mano"]["mano_path"],
        run_cfg["train"]["loss"],
        use_pc=True,
    ).to(device)
    diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine")
    schedule_sampler = create_named_schedule_sampler(name="uniform", diffusion=diffusion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    # resume optimizer
    if (reload_ckpt_optimizer_filepath := run_cfg["train"]["reload_ckpt_optimizer_filepath"]) is not None:
        _logger.info("optimzier reload ckpt: %s", reload_ckpt_optimizer_filepath)
        reload_ckpt_optimizer_param = torch.load(reload_ckpt_optimizer_filepath, map_location=device)
        optimizer.load_state_dict(reload_ckpt_optimizer_param)
    milestones = [epoch - now_epoch for epoch in run_cfg["train"]["scheduler_milestone"]]
    for i in range(len(milestones)):
        if milestones[i] == 0:
            milestones[i] = -1
    _logger.info("update scheduler_milestone: %s", milestones)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=run_cfg["train"]["scheduler_gamma"],
    )

    # summary writer
    summary_writer_dir = ckpt.handle_save_path(f"?(ckpt_path)/summary", ckpt_path=ckpt_path) if commit else None
    summary_writer = DDPSummaryWriter(log_dir=summary_writer_dir, rank=rank)

    # seed
    seed = run_cfg["runtime"]["seed"]
    if rank is not None:
        seed = seed + rank
    random_util.setup_seed(seed)

    # mainloop
    dist.barrier()
    num_epoch = run_cfg["train"]["num_epoch"]
    for epoch_id in range(now_epoch, num_epoch):
        # epoch begin
        train_sampler.set_epoch(epoch_id)

        # epoch iterate
        pbar = (
            tqdm.tqdm(
                total=len(train_dataloader), position=0, bar_format=pbar_util.fmt, desc=f"train epoch {epoch_id:>04d}:"
            )
            if not rank
            else pbar_util.dummy_pbar()
        )
        for batch_id, batch in enumerate(train_dataloader):
            step_id = (epoch_id * len(train_dataloader) + batch_id) * world_size

            optimizer.zero_grad()
            model.train()

            batch_device = map_copy_select_to(
                batch,
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

            t, weights = schedule_sampler.sample(batch_size, device)
            x_start = batch_device["pose_repr"]  # (bs, seqlen, in_dim)
            x_start = x_start.unsqueeze(3)  # (bs, seqlen, in_dim, 1)
            x_start = x_start.permute((0, 2, 3, 1))  # (bs, in_dim, 1, seqlen)
            loss_store, (extra_loss, extra_loss_store) = diffusion.training_losses(
                model, x_start, t, model_kwargs={"batch": batch_device}, loss_callback=model_extra_loss
            )
            loss = (loss_store["loss"] * weights).mean()
            loss_store["diffusion_loss"] = loss
            loss = loss + extra_loss

            loss.backward()
            clip_gradient(optimizer, 0.1, 2.0)
            optimizer.step()
            optimizer.zero_grad()

            # summary writer
            summary_extra_loss(
                summary_writer,
                loss_store,
                extra_loss_store,
                lr=float(next(iter(optimizer.param_groups))["lr"]),
                step_id=step_id,
            )

            pbar.update()
        pbar.close()

        # epoch end
        scheduler.step()
        dist.barrier()
        _logger.info("train epoch %04d conclude | loss: %f", epoch_id, loss.item())
        _logger.info("train epoch %04d lr %s", epoch_id, [group["lr"] for group in optimizer.param_groups])
        _logger.info("train epoch %04d detail loss:", epoch_id)
        _logger.info("                     %s: %f", "diffusion_loss".ljust(20), float(loss_store["diffusion_loss"]))
        _logger.info("                     %s: %f", "MoE_loss".ljust(20), sum(model.module.aux_loss_list))
        _logger.info("                     %s: %f", "MoE_loss_for_timestep".ljust(20), sum(model.module.aux_loss_list_t))
        _logger.info("                     %s: %f", "MoE_loss_for_action".ljust(20), sum(model.module.aux_loss_list_a))
        for loss_name, loss_node in extra_loss_store.items():
            if loss_name in ["loss"]:
                continue
            _logger.info("                     %s: %f", loss_name.ljust(20), float(loss_node))

        if not rank:
            # record_state
            record_freq = run_cfg["train"]["record_freq"]
            if (
                commit
                and (record_freq is not None and record_freq != -1)
                and (epoch_id == 0 or epoch_id % record_freq == record_freq - 1 or epoch_id == num_epoch - 1)
            ):
                model_weight_path = ckpt.handle_save_path(
                    f"?(ckpt_path)/save/model_{epoch_id:0>4}.pt", ckpt_path=ckpt_path
                )
                save_state(model.state_dict(), model_weight_path, remove_prefix="module", filter_out=["clip_model", "long_clip_model"])
                optimizer_weight_path = ckpt.handle_save_path(
                    f"?(ckpt_path)/save/optimizer_{epoch_id:0>4}.pt", ckpt_path=ckpt_path
                )
                save_state(optimizer.state_dict(), optimizer_weight_path)

        dist.barrier()

    # conclude
    ddp_util.destroy_ddp()


def main():
    config_reg = ConfigRegistry(prog=PROG)
    ckpt.reg_entry(config_reg)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    ckpt_cfg = ckpt.reg_extract(config_reg)
    run_cfg = reg_extract(config_reg)

    device_id_list = run_cfg["runtime"]["device_id"]
    ddp_util.validate_device_id_list(device_id_list)
    world_size = len(device_id_list)
    # replace underlying device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(el) for el in device_id_list)
    run_cfg["runtime"]["device_id"] = list(range(world_size))

    mp.spawn(run, args=(world_size, ckpt_cfg, run_cfg), nprocs=world_size)


if __name__ == "__main__":
    main()
