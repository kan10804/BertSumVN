# script_train_model.py
import os
import json
import shutil
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.init import xavier_normal_
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel

from transformer_model import BertAbsSum
from transformer_preprocess import DataProcessor
from transformer_utils import *
from params_helper import Constants, Params

set_seed(0)
logger = get_logger(__name__)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if Params.visible_gpus == "-1":
    Params.visible_gpus = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus

num_gpus = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained(Params.bert_model, local_files_only=False)


def init_process(rank, world_size):
    if world_size <= 1:
        return
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = Params.ddp_master_port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def load_checkpoint_if_available(checkpoint_dir, device):
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, "Best_Checkpoint.pt")
    if os.path.exists(path):
        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, map_location=device)
    return None


def get_model(rank, device, checkpoint, output_dir):
    logger.info(f"*** Getting model at rank {rank} ***")

    if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir and checkpoint is None:
        # load config from resume dir if present
        config_path = os.path.join(Params.resume_checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = None
    else:
        bert_cfg = AutoModel.from_pretrained(Params.bert_model).config
        config = {
            "bert_model": Params.bert_model,
            "bert_config": bert_cfg.__dict__,
            "decoder_config": {
                "vocab_size": bert_cfg.vocab_size,
                "d_word_vec": bert_cfg.hidden_size,
                "n_layers": Params.decoder_layers_num,
                "n_head": bert_cfg.num_attention_heads,
                "d_k": Params.decoder_attention_dim,
                "d_v": Params.decoder_attention_dim,
                "d_model": Params.decoder_attention_dim,  # ensure you set this in Params appropriately
                "d_inner": bert_cfg.intermediate_size
            },
            "freeze_encoder": Params.freeze_encoder
        }

    # save config on main process
    if rank == 0 and config is not None:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, default=str)

    model = BertAbsSum(config=config, constants=Constants, device=device)
    if checkpoint:
        # checkpoint may have 'model' or direct state_dict
        ck = checkpoint
        if "model" in ck:
            model_state = ck["model"]
        elif "model_state_dict" in ck:
            model_state = ck["model_state_dict"]
        else:
            model_state = ck
        # load state dict leniently
        model.load_state_dict(model_state, strict=False)

    model.to(device)

    if num_gpus > 1:
        # wrap model with DDP after moving to GPU
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    return model


def get_optimizer(model, checkpoint):
    params_iter = model.module.named_parameters() if isinstance(model, DistributedDataParallel) else model.named_parameters()
    model_params = list(params_iter)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model_params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    opt = torch.optim.AdamW(groups, lr=Params.learning_rate)
    if checkpoint and "optimizer_state_dict" in checkpoint:
        try:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            logger.warning("Could not load optimizer state - skipping.")
    return opt


def get_train_dataloader(rank, world_size):
    logger.info(f"Loading train data at {Params.train_data_path}")
    data = torch.load(Params.train_data_path)
    proc = DataProcessor()
    loader = proc.create_distributed_dataloader(rank, world_size, data, Params.train_batch_size) if world_size > 1 else proc.create_dataloader(data, Params.train_batch_size)
    check_data(loader)
    return loader


def get_valid_dataloader(rank, world_size):
    logger.info(f"Loading valid data at {Params.valid_data_path}")
    data = torch.load(Params.valid_data_path)
    loader = DataProcessor().create_dataloader(data, Params.valid_batch_size, shuffle=False)
    check_data(loader)
    return loader


def check_data(loader):
    logger.info("*** Checking data ***")
    batch = next(iter(loader))
    try:
        src_ids = batch[1]
        tgt_ids = batch[3]
        logger.info(f"Source: {tokenizer.decode(src_ids[0], skip_special_tokens=True)}")
        logger.info(f"Target: {tokenizer.decode(tgt_ids[0], skip_special_tokens=True)}")
    except Exception as e:
        logger.warning("Unable to decode sample for logging: " + str(e))


def cal_performance(logits, ground):
    # ground: (B, T) with BOS at [:,0]; shift as needed
    ground = ground[:, 1:]  # predict from token 1 onward
    logits = logits[:, :-1, :]  # align logits (B, T-1, V)
    logits = logits.contiguous().view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)
    loss = F.cross_entropy(logits, ground, ignore_index=Constants.PAD, label_smoothing=Params.label_smoothing_factor)
    pad = ground.ne(Constants.PAD)
    pred = logits.max(-1)[1]
    correct = pred.eq(ground).masked_select(pad).sum().item()
    tokens = pad.sum().item()
    return loss, correct, tokens


def init_parameters(model):
    # only initialize parameters of decoder / projection / linear (not encoder if pretrained)
    for n, p in model.named_parameters():
        if "encoder" not in n and p.dim() > 1:
            xavier_normal_(p)


def validate(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for batch in valid_loader:
            src = batch[1].to(device)
            src_mask = batch[2].to(device)
            tgt = batch[3].to(device)
            tgt_mask = batch[4].to(device) if len(batch) > 4 else None

            logits = model(batch_src_seq=src, batch_src_mask=src_mask, batch_tgt_seq=tgt, batch_tgt_mask=tgt_mask)
            loss, correct, tokens = cal_performance(logits, tgt)
            total_loss += loss.item() * tokens
            total_correct += correct
            total_tokens += tokens
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, acc


def train(rank, world_size, output_dir):
    if world_size > 1:
        init_process(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Rank {rank}/{world_size} process initialized on {device}.")

    train_loader = get_train_dataloader(rank, world_size)
    valid_loader = get_valid_dataloader(rank, world_size)

    # resume checkpoint if provided
    checkpoint = None
    if Params.resume_checkpoint_dir:
        checkpoint = load_checkpoint_if_available(Params.resume_checkpoint_dir, device)

    # DDP barrier so all processes wait here (if multi-GPU)
    if world_size > 1:
        dist.barrier()

    model = get_model(rank, device, checkpoint, output_dir)

    # Only init parameters if no checkpoint loaded (don't clobber pretrained weights)
    if checkpoint is None:
        init_parameters(model)

    optimizer = get_optimizer(model, checkpoint)

    model.train()
    global_step = 0

    for epoch in range(1, Params.num_train_epochs + 1):
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", ascii=True)
        for step, batch in enumerate(train_iter, start=1):
            global_step += 1
            src = batch[1].to(device)
            src_mask = batch[2].to(device)
            tgt = batch[3].to(device)
            tgt_mask = batch[4].to(device) if len(batch) > 4 else None

            logits = model(batch_src_seq=src, batch_src_mask=src_mask, batch_tgt_seq=tgt, batch_tgt_mask=tgt_mask)
            loss, _, _ = cal_performance(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                train_iter.set_postfix({"Loss": loss.item()})

        # End epoch: run validation on main process only
        if rank == 0:
            val_loss, val_acc = validate(model, valid_loader, device)
            logger.info(f"Epoch {epoch} — Val Loss: {val_loss:.4f}, Val token-acc: {val_acc:.4f}")
            # save checkpoint each epoch (you can add best-checkpoint logic)
            save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt")
            checkpoint_to_save = {
                "model": model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
                "model_config": model.module.config if isinstance(model, DistributedDataParallel) else model.config,
                "model_arguments": {
                    "max_src_len": getattr(Params, "max_src_len", None),
                    "max_tgt_len": getattr(Params, "max_tgt_len", None),
                    "decoder_layers": getattr(Params, "decoder_layers_num", None)
                },
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint_to_save, save_path)
            logger.info(f"Saved checkpoint → {save_path}")

    # final save on main process
    if rank == 0:
        final_save = os.path.join(output_dir, "Best_Checkpoint.pt")
        checkpoint_to_save = {
            "model": model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
            "model_config": model.module.config if isinstance(model, DistributedDataParallel) else model.config,
            "model_arguments": {
                "max_src_len": getattr(Params, "max_src_len", None),
                "max_tgt_len": getattr(Params, "max_tgt_len", None),
                "decoder_layers": getattr(Params, "decoder_layers_num", None)
            },
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint_to_save, final_save)
        logger.info(f"Saved FULL checkpoint → {final_save}")

    if world_size > 1:
        dist.destroy_process_group()


def cleanup_on_error(out):
    if os.path.isdir(out) and len(os.listdir(out)) < 2:
        shutil.rmtree(out)


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
    normalized = Params.bert_model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    model_dir = f"model_{normalized}_{Params.decoder_layers_num}layers_{timestamp}"
    output_dir = os.path.join(Params.output_dir, model_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        if WORLD_SIZE == 1:
            train(0, 1, output_dir)
        else:
            mp.spawn(train, args=(WORLD_SIZE, output_dir), nprocs=WORLD_SIZE)
    except Exception as e:
        logger.error("Training failed!")
        cleanup_on_error(output_dir)
        raise e
