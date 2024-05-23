import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
import copy
import random
import time
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
from ttts.vqvae.dataset import VQGANDataset, VQVAECollater, BucketSampler, DistributedBucketSampler
from typing import List, Optional, Tuple, Union
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
from ttts.vqvae.rvq1 import RVQ1
from ttts.vqvae.vq3 import SynthesizerTrn
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from ttts.utils import commons
import torchaudio
from ttts.vqvae.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from ttts.vqvae.hifigan import MultiPeriodDiscriminator
from torchaudio.functional import phase_vocoder, resample, spectrogram
cfg_path = 'ttts/vqvae/config_v3.json'
cfg = json.load(open(cfg_path))
hps = HParams(**cfg)
# os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist, traceback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import logging, traceback

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from random import randint

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
global_step = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
def main(model_path=None):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    print(f"ngpus:{n_gpus}")
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
            model_path,
        ),
    )
def get_state_dict(model):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict
def load_state_dict(model, new_state_dict, strict=True):
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict,strict=strict)
    else:
        model.load_state_dict(new_state_dict,strict=strict)
def save(G, D, G_opt, D_opt, epoch, milestone, logs_folder):
    global global_step
    data = {
        'step': global_step,
        'epoch': epoch,
        'G': get_state_dict(G),
        'D': get_state_dict(D),
        'G_opt': get_state_dict(G_opt),
        'D_opt': get_state_dict(D_opt)
    }
    torch.save(data, str(logs_folder / f'model-{milestone}.pt'))
def load(model_path, device, G, D, G_opt, D_opt):
    global global_step
    data = torch.load(model_path, map_location=device)
    G_state_dict = data['G']
    D_state_dict = data['D']
    G_opt_state_dict = data['G_opt']
    D_opt_state_dict = data['D_opt']
    global_step = data['step']
    epoch = data['epoch']
    current_model_dict = get_state_dict(G)
    G_state_dict={k:v if v.size()==current_model_dict[k].size()
        else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), G_state_dict.values())}
    load_state_dict(G, G_state_dict, strict=False)
    load_state_dict(D, D_state_dict)
    try:
        G_opt.load_state_dict(G_opt_state_dict)
    except:
        print('Fail to load G_opt')
    D_opt.load_state_dict(D_opt_state_dict)
    return epoch

def run(rank, n_gpus, hps, model_path):
    global global_step
    if rank == 0:
        now = datetime.now()
        logs_folder = Path(hps.train.logs_folder+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
        logs_folder.mkdir(exist_ok = True, parents=True)
        writer = SummaryWriter(log_dir=logs_folder)

    dist.init_process_group(
        backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    train_dataset = VQGANDataset(hps)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = VQVAECollater()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=16,
    )

    G = SynthesizerTrn(hps.data.filter_length // 2 + 1,hps.train.segment_size // hps.data.hop_length, **hps.vqvae)
    D = MultiPeriodDiscriminator()
    if rank==0:
        print("G params:", count_parameters(G))
        print("D params:", count_parameters(D))
    G = G.cuda(rank)
    D = D.cuda(rank)

    G_optimizer = AdamW(G.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    D_optimizer = AdamW(D.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    G = DDP(G, device_ids=[rank], find_unused_parameters=True)
    D = DDP(D, device_ids=[rank], find_unused_parameters=True)

    device = f"cuda:{rank}"
    if model_path is not None:
        epoch_str = load(model_path,device,G,D,G_optimizer,D_optimizer)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        G_optimizer, gamma=hps.train.lr_decay, last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        D_optimizer, gamma=hps.train.lr_decay, last_epoch=-1
    )
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                logs_folder,
                epoch,
                hps,
                [G, D],
                [G_optimizer, D_optimizer],
                [scheduler_g, scheduler_d],
                [train_loader, None],
                [writer, None],
            )
        else:
            train_and_evaluate(
                rank,
                None,
                epoch,
                hps,
                [G, D],
                [G_optimizer, D_optimizer],
                [scheduler_g, scheduler_d],
                [train_loader, None],
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, logs_folder, epoch, hps, nets, optims, schedulers, loaders, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, _ = loaders
    if writers is not None:
        writer, _ = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, data in tqdm(enumerate(train_loader)):
        wav = data['wav'].cuda(rank, non_blocking=True).detach()
        wav_length = data['wav_lengths'].cuda(rank, non_blocking=True).detach()
        text = data['text'].cuda(rank, non_blocking=True).detach()
        text_length = data['text_lengths'].cuda(rank, non_blocking=True).detach()
        lang = data['langs'].cuda(rank, non_blocking=True).detach()

        spec = spectrogram_torch(wav, hps.data.filter_length,
            hps.data.hop_length, hps.data.win_length, center=False).squeeze(0)
        spec_length = torch.LongTensor([
            x//hps.data.hop_length for x in wav_length]).cuda(rank, non_blocking=True)

        (y_hat, ids_slice, l_length, l_detail, l_text_detail, l_dur_detail, z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q, m_t, logs_t),
            stats_ssl,) = net_g(spec, spec_length, text, text_length, lang)

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

        y = commons.slice_segments(
            wav.unsqueeze(1), ids_slice * hps.data.hop_length, hps.train.segment_size
        )  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc
        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

        loss_detail = l_detail
        loss_text_detail = l_text_detail
        loss_dur_detail = l_dur_detail
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss_kl_text = kl_loss(z_p, logs_q, m_t, logs_t, z_mask)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel \
            + loss_kl + loss_kl_text + loss_dur \
            + loss_detail + loss_text_detail + loss_dur_detail

        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

        if rank == 0:
            if global_step % hps.train.val_freq == 0:
                lr = optim_g.param_groups[0]["lr"]
                net_g.eval()
                with torch.no_grad():
                    wav_eval = net_g.module.infer(text, text_length, spec, spec_length, lang)
                net_g.train()
                scalar_dict = {
                        "gen/loss_gen_all": loss_gen_all,
                        "gen/loss_gen":loss_gen,
                        'gen/loss_fm':loss_fm,
                        'gen/loss_mel':loss_mel,
                        'gen/loss_dur':loss_dur,
                        'gen/loss_detail':loss_detail, 
                        'gen/loss_text_detail':loss_text_detail,
                        'gen/loss_dur_detail':loss_dur_detail,
                        'gen/loss_kl':loss_kl, 
                        'gen/loss_kl_text':loss_kl_text,
                        "norm/G_grad": grad_norm_g, 
                        "norm/D_grad": grad_norm_d,
                        'disc/loss_disc_all':loss_disc_all,
                        'gen/lr':lr,
                    }
                image_dict = {
                    "img/mel": plot_spectrogram_to_numpy(y_mel[0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                    "img/mel_pred": plot_spectrogram_to_numpy(y_hat_mel[0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                    "img/mel_raw": plot_spectrogram_to_numpy( mel[0].data.cpu().numpy()),
                    "img/stats_ssl": plot_spectrogram_to_numpy(stats_ssl[0].data.cpu().numpy()),
                }
                audios_dict = {
                    'wav/gt':wav[0].detach().cpu(),
                    'wav/pred':wav_eval[0].detach().cpu()
                }
                milestone = global_step // cfg['train']['save_freq'] 
                torchaudio.save(str(logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), hps.data.sampling_rate)
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                    audios=audios_dict,
                    audio_sampling_rate=hps.data.sampling_rate
                )
        if global_step % hps.train.save_freq == 0 and rank == 0:
            keep_ckpts = hps.train.keep_ckpts
            if keep_ckpts > 0:
                clean_checkpoints(path_to_models=logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
            save(net_g, net_d, optim_g, optim_d, epoch, milestone, logs_folder)
        global_step += 1
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))

if __name__ == "__main__":
    model_path = None
    model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-05-19-18-02-08/model-309.pt'
    main(model_path=model_path)
