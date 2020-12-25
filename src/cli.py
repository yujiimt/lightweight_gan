import os
import fire
import retry.api import retry_call
from tqdm import tqdm
from datatime import datetime 
from functools import wraps
from lightweight_gan import Trainer, NanException
from lightweight_gan.diff_augment_test import DiffAugmentTest

import torch 
import torch.multiprocessing as mp
import totch.distributed as dist 

import numpy as np


def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = "generated-"):
    now = datetime.now()
    timestamp = now.strtime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(sedd)

def run_traininig(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group('nccl', rank=rank, world_size = world_size)

        print(f"{rank + 1} / {world_size} process initialized .")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    
    model.set_data_src(data)

    for _ in tqdm(num_train_steps - model.steps), initial = model.steps, total = num_train_steps, miniterval = 10., desc = f'{name}<{data}>'):
        retry_call(model.train, tries = 3, exception = NanException)
        if is_main and _ % 50 == 0:
            model.print_log()
    
    model.save(model.check_point_num)

    if is_ddp:
        dist.destroy_process_group()


def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 256,
    optimizer = 'adam',
    fmap_max = 512,
    transparent = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    generated_interpolation = False,
    aug_test = False,
    attn_res_layers = [32],
    sle_spatial = False,
    disc_output_size = 1,
    anatialias = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_titles = 8,
    trunc_psi = 0.75,
    aug_prob = None,
    aug_types = ["cutout", "translation"],
    dataset_aug_prob = 0.,
    multi_gpus = False,
    calculate_fid_every = None,
    seed = 42,
    amp = False
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        attn_res_layers = cast_list(attn_res_layers),
        sle_spatial = sle_spatial,
        disc_output_size = disc_output_size,
        anatialias = anatilias,
        image_size = image_size, 
        optimizer = optimizer,
        fmap_max = fmap_max,
        transparent = transparent,
        greyscale = greyscale,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        trunc_psi = trunc_psi,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        amp = amp        
    )

    if generate:
        model = Treainer(**model_args)
        model.load(load_from)
        saples_names = timestamped_filename()
        model.evaluate(samples_name, num_image_tiles)
        print(f'samples images generated at {results_dir}/{name}/{samples_name}')
        return
    
    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save)

