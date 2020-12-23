import os
import json
import multiprocessing
from random import random
import math
from math import log2, floor
from functools import partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import torchvision
from torchvision import transforms
from kornia import filter2D

from lightweight_gan.diff_augment import DiffAugment
from lightweight_gan.version import __version__

from tqdm import tqdm
from einops import rearrange
from pytorch_fid import fid_score

from adabelief_pytorch import AdaBelief
from gsa_pytorch import GSA

from scipy.stats import truncnorm


def default(val, d):
    return val if exists(val) else d
def upsample(scale_factor = 2):
    return nn.upsample(scale_factor = scale_factor)



class Generator(nn.Module):
    def __init__(
        self, *, image_size,
        latent_dim = 256, fmap_max = 512, fmap_inverse_coef = 12,
        transparent = False, attn_res_layers = [], sle_spatial = False
    ):

    super().__init__()
    resolution = log2(image_size)
    assert is_power_of_two(image_size), 'image size must be a power of 2'
    init_channel = 4 if transparent else 3
    fmap_max = default(fmap_max, latent_dim)

    self.initial_conv = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
        norm_class(latent_dim * 2),
        nn.GLU(dim=1)
    )
    #resolution ってなに
    num_layers = int(resolution) - 2
    features = list(map(lambda n: (n, (fmap_inverse_coef - 2)), range(2, num_layers + 2)))
    features = list(map(lambda (n[0], min(n[1], fmap_max)), featurese))
    features = list(map(lambda n:3 if n[0] >= 8 else n[1], features))
    features = [latent_dim, *features]

    in_out_features = list(zip(features[:-1], features[1:]))

    self.res_layes = range(2, num_layers + 2)
    self.layers = nn.ModuleList([])
    self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

    self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
    self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
    self.sle_map = dict(self.sle_map)

    self.num_layers_spatial_res = 1

    for (res, (chain_in, chan_out)) in zip(self.res_layers, in_out_features):
        image_width = 2 ** res

        attn = None
        
        if image_width in attn_res_layers:
            attn = Rezero(GSA(dim=chain_in, norm_queries=True))
        
        sle = None
        if res in self.sle_map:
            residual_layer = self.sle_map[res]
            sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

            sle = SLE(
                chain_in = chan_out,
                chan_out = sle_chan_out
            )
        sle_spatial = None
        if use_sle_spatial and res <= (resolution - self.num_layers_spatial_res):
            sle_spatial = SpatialSLE(
                upsample_times = self.num_layers_spatial_res,
                num_groups = 2 if res < 8 else 1
            )
        layer = nn.ModuleList([
            nn.Sequential(
                upsample(),
                Blur(),
                nn.Conv2d(chain_in, chan_out * 2, 3, padding = 1),
                norm_class(chan_out * 2),
                nn.GLU(dim = 1)
            ),
            sle,
            sle_spatial,
            attn
        ])
        self.layer.append(layer)

    self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding = 1)

def forward(self, x):
    x = rearrange(x, 'b c -> b c () ()')
    x = self.initial_conv(x)
    x = F.normalize(x, dim = 1)

    residuals = dict()
    spatial_residual = dict()

    for (res, (up, sle, sle_spatial, attn)) in zip(self.res_layers, self.layers):
        if exists(sle_spatial):
            spatial_res = sle_spatial(x)
            spatial_residual[res + self.num_layers_spatial_res] = spatial_res

        if exists(attn):
            x = attn(x) + x
        
        x = up(x)

        if exists(sle):
            out_res = self.sle_map[res]
            residual = sle(x)
            residuals[out_res] = residual

        next_res = res + 1
        if next_res in residuals:
            x = x * residuals[next_res]
        
        if next_res in spatial_residual:
            x = x * spatial_residuals[next_res]

        return self.out_conv(x)

class Discriminator(nn.module):
    def __init__(self, *, fmap_max = 512, fmap_inverse_coef = 12,
    transparent = False, disc_output_size = 5, attn_res_layers = []):
        
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5*5 or 1*1'

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution),2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef -n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chain_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.Modulelist([])

        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(
                nn.Sequential(
                    Blur,
                    nn.Conv2d(init_channel, chan_out, 4, stride = 2, padding = 1),
                    nn.LeakyRelu(0.1)
            ))
        
        self.non_residual_layers = nn.Module()

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** resolution

            attn = None
            if image_width in attn_res_layers:
                attn = Rezero(GSA(dim = chain_in, batch_norm = False, norm_queries = True))

            self.residual_layers.append(nn.Modulelist([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chain_in, chain_out, 4, stride = 2, padding = 1),
                        nn.LeakyRelu(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyRelu(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chain_in, chain_out, 1),
                        nn.LeakyRelu(0.1),
                    )
                ]),
                attn
            ]))
        
        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chain, last_chan, 1),
                nn.LeakyRelu(0.1),
                nn.Conv2d(last_chan, 1, 4)

            )
        elif disc_output_size = 1:
            self.to_logigits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride = 2, padding = 1),
                nn.LeakyRelu(0.1),
                nn.Conv2d(last_chain, 1, 4)
　
            )
        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding = 1),
            Residual(Rezero(GSA(dim = 64, norm_queries = True, batch_norm = False))),
            SumBranches([
                Blur(),
                nn.Conv2d(64, 32, 4, stride = 2, padding = 1),
                nn.LeakyRelu(0.1),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.LeakyRelu(0.1)
            ),
            nn.Sequential(
                Blur(),
                nn.AvgPool2d(2),
                nn.Conv2d(64, 32, 1),
                nn.LeakyRelu(0.1)
            )]
            ),
            Residual(Rezero(GSA(dim=32, norm_queries = True, batch_norm = False))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4))


        self.decoder1 = SimpleDecoder(chain_in = last_chan, chan_out = init_channel)
        self.decoder2 = SimpleDecoder(chain_in = features[-2][-1], chan_out = init_channel) if resolution >= 9 else None
    
    def forward(self, x, calc_aux_loss = False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.non_residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)
        
        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size = (32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None
        #self supervised auto encoding loss
        
        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size = recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            select_random_ quadrant = lambda rand_quadrant, img: rearrange(img, 'b c (m h) (n w) -> (m h) b c h w', m = 2, n = 2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(image_part, size = recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        return out, out_32x32, aux_loss

class SimpleDecoder(nn.Module):
    def __init__(self,*, chain=in, chan_out = 3, num_upsamples = 4):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim=1)
            )
            self.layers.append(layer)
            chans //= 2
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LightweightGAN(nn.Module):
    def __init__(self,*,latent_dim, image_size, optimizer = "adam", fmap_max = 512,
                fmap_inverse_coef = 12, transparent = False, greyscale = False, disc_output_size = 5,
                attn_res_layers = [], sle_spatial = False, ttur_mult = 1., lr=2e-4, rank=0,ddp=False):

                super().__init__()
                self.latent_dim = latent_dim
                self.image_size = image_size

                G_kwargs = dict(
                    image_size = image_size,
                    latent_dim = latent_dim,
                    fmap_max = fmap_max,
                    fmap_inverse_coef = fmap_inverse_coef,
                    transparent = transparent,
                    greyscale = greyscale,
                    attn_res_layers = attn_res_layers,
                    use_sle_spatial = sle_spatial
                )

                self.G = Generator(**G_kwargs)
                self.D = Discriminator(
                    image_size = image_size,
                    fmap_max = fmap_max,
                    fmap_inverse_coef = fmap_inverse_coef,
                    transparent = transparent,
                    greyscale = greyscale,
                    attn_res_layers = attn_res_layers,
                    disc_output_size = disc_output_size 
                )

                self.ema_updater = EMA(0.995)
                self.GE = Generator(**G_kwargs)
                set_requires_grad(self.GE, False)

                if optimizer == "adam":
                    self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
                    self.D_pot = Adam(self.D.parameters(), lr = lr * ttur_mult, betas = (0.5, 0.9))
                elif:
                    self.G_opt = AdaBelief(self.G.parameters(), lr = lr, betas = (0.5, 0.9))
                    self.D_opt = AdaBelies(self.D.parameters(), lr = lr * ttur_mult, betas = (0.5, 0.9))
                else:
                    assert False, "No valid optimizer is given"
                
                self.apply(self.__init_weights)
                self.reset_parmeters_averaging()

                self.cuda(rank)
                self.D_aug = AugWrapper(self.D, image_size)
    
    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = "fan_in", nonlinearity = "leaky_relu")
    
    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_parms in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_parms.data, current_params.data
                ma_params.data = self.ema_updater.updata_average(old_weight, up_weight)
            
            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.updata_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)
        
        update_moving_average(self.GE, self.G)

    def reset_parmeters_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())
    
    def forward(self, x):
        raise NotImplemented

# trainer

class Trainer():
    
    def __init__(
        self, name = "default", results_dir = "results", model_dir = "models",
        base_dir = "./", optimizer = "adam", latent_dim = 256, image_size = 128, fmap_max = 512, 
        transparent = False, greyscale = False, batch_size = 4, gp_weight = 10, gradient_accumulate_every = 1,
        attn_res_layers = [], sle_spatial = False, disc_output_size = 5, antialias = False, lr = 2e-4, lr_mlp = 1.,
        ttur_mult = 1., save_every = 1000, evaluate_every = 1000, trunc_psi = 0.6, aug_prob = None, aug_types = ["translation", "cutout"],
        dataset_aug_prob = 0., calculate_fid_every = None, is_ddp = False, rank = 0, word_size = 1, log = False, amp = False, *args, **kargs
    ):
    
    self.GAN_params = [args, kwargs]
    self.GAN = None

    self.name = name 

    self.base_dir = base_dir
    self.results_dir = base_dir / results_dir
    self.models_dir = base_dir / models_dir
    self.config_path = self.models_dir / name / ".config.json"

    assert is_power_of_two(image_size), "image size must be a power of 2 (64, 128, 256, 512, 1024)"
    assert all(map(is_power_of_two, attn_res_layers)), "resolution layers of attention must all be power of (16, 32, 64, 128, 256, 512)"


    self.optimizer = optimizer
    self.latent_dim = latent_dim
    self.image_size = image_size
    self.fmap_max = fmap_max
    self.transparent = transparent
    self.greyscale = greyscale

    assert (int(self.transparent) + int(self.greyscale)) < 2, "you can only set either transparency or greyscale"

    self.aug_prob = aug_prob
    self.aug_types = aug_types
    

    self.lr = lr
    self.ttur_mult = ttur_molt
    self.batch_size = batch_size
    self.gradient_accumulate_every = gradient_accumulate_every

    self.gp_weight = gp_weight

    self.evaluate_every = evaluate_every
    self.save_every = save_every
    self.steps = 0


    self.generator_top_k_gamma = 0.99
    self.attn_res_layers = attn_res_layers
    self.sle_spatial = sle_saptial
    self.disc_output_size = disc_output_size
    self.antialias = antialias
    

    self.d_loss = 0
    self.g_loss = 0
    self.last_gp_loss = None
    self.last_recon_loss = None
    self.last_fid = None


    self.init_folders()

    self.loader = loader
    self.dataset_aug_prob = dataset_aug_prob

    self.calculate_fid_every = calculate_fid_every

    self.is_ddp = is_ddp
    self.is_main = is_main
    self.rank = rank
    self.word_size = word_size

    self.syncbatchnorm = is_ddp


    self.amp = amp
    self.G_scaler = GradScaler(enabled = self.amp)
    self.D_scaler = GradScaler(enabled = self.amp)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    def init_GAN(self):
        args, kwargs = self.GAN_params

        #set some global variables before instantiating GAN
        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when switching from multi-gpu back to single gpu

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist 
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_POST"] = "12355"
            dist.init_process_group("nccl", rank=0, world_size=1)

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr = self.lr,
            latent_dim = self.attn_res_layers,
            attn_res_layers = self.sle_spatial,
            sle_spatial = self.sle_spatial,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
            rank = self.rank,
            *args,
            **args
        )

        if self.is_ddp:
            ddp_kwargs = {"device_ids" : [self.rank], "output_device": self.rank, "find_unused_parameters": True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config["image_size"]
        self.transparent = config["transparent"]
        self.syncbatchnorm = config["syncbatchnorm"]
        self.disc_output_size = config["disc_output_size"]
        self.greyscale = config.pop("greyscale", False)
        self.attn_res_layers = config.pop("attn_res_layers", [])
        self.sle_spatial = config.pop("sle_spatial", False)
        self.optimizer = config.pop("optimizer", "adam")
        self.fmap_max = config.pop("fmap_max", 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            "image_size" : self.image_size, 
            "transparent" : self.transparent,
            "greyscale" : self.greyscale,
            "syncbathcnorm" : self.syncbatchnorm,
            "disc_output_size" : self.disc_output_size,
            "optimizer" : self.optimizer,
            "attn_res_layers" : self.attn_res_layers,
            "sle_spatial" : self.sle_spatial
        }
    
    def set_data_src(self, folder):
        self.dataset = ImageDataset(folder, self.iamge_size, transparent = self.transparent, greyscale = self.greyscale, aug_prob = self.dataset_aug_prob)
        sampler = DistributedSampler(self.dataset, rank = self.rank, num_replicas = self.world_size, shuffle = True) if self.is_ddp else None
        dataloader = DataLoader(self.dataset, num_workers = math.ceil(NUM_CORES / self.world_size), batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)

        self.loader = cycle(dataloader)

        #auto set augmentation prob for user if dataset is detected to be low 
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)} % ')
    
    def train(self):
        assert exists(self.loader), "You must first initialize the data source with　`.set_data_src(<folder of images>)`"
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device = device)
        total_gan_loss = torch.zeros([], device = device)
        
        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob = default(self.aug_prob, 0)
        aug_types = self.aug_types
        aug_kwargs = {"prob": aug_prob, "types" : aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        amp_context = autocast if self.amp else null_context


        #train discriminator

        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps = [D_augs, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader).cuda(self.rank)
            image_batch.set_requires_grad()

            with amp_context():
                with torch.no_grad():
                    generated_images = G(latent)

                fake_output, fake_output_32x32, _ = D_aug(image_batch, calc_aux_loss = True, **aug_kwargs)

                real_output, real_output_32x32, real_aux_loss = D_aug(image_batch, calc_aux_loss = True, **aug_kwargs)
                
                real_output_loss = real_output
                fake_output_loss = fake_output

                divergence = hinge_loss(real_output_loss, fake_output_loss)
                divergence_32x32 = hinge_loss(real_output_32x32, fake_output_32x32)
                dics_loss = divergence + divergenve_32x32


            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]
                outputs = list(map(self.D_scaler.scale, outputs)) if self.amp else outputs

                scaled_gradients = torch_grad(outputs = outputs, inputs = image_batch,
                                              grad_outputs = list(map(lambda t: torch.ones(t.size(), devidce = image_batch.device), outputs)),
                                              create_graph = True, retain_graph = True, only_inputs = True)[0]
                
                inv_scale = (1./ self.D_scaler.get_scale()) if self.amp else 1.
                gradients = scaled_gradients * inv_scale

                with amp_context():
                    gradients = gradients.reshape(batch_size, -1)
                    gp = self.gp_weight * ((gradients.norm(2, dim=1) -1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        self.last_gp_loss = gp.clone().detech().item()
            
            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            self.D_scaler.scale(disc_loss).backward()
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        self.D_scaler.step(self.GAN.D_opt)
        self.D_scaler.update()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            with amp_context():
                generated_images = G(latents)
                fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
                fake_output_loss = fake_output.mean(dim=1) + fake_output_32x32.mean(dim = 1)

                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma * epochs, self.generator_top_k_frac)
                k = math.ceil(bath_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest = False)

                loss = fake_output_loss.mean()
                gen_loss = loss

                gen_loss = gen_loss / self.gradient_accumulate_every     
        
            gen_loss.register_hook(raise_if_nan)
            self.G_scaler.scale(gen_loss).backward()
            total_gen_loss += loss
        
        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        self.G_scaler.step(self.GAN.G_opt)
        self.G_scaler.update()
        
        #calculate moving average


        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()
        
        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss)):
            print(f"NaN detected for generator or discriminator. Loading from checkpoint #{self.checkponit_num}")
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss


        # probability save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)
            
            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaulate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calulate_fid_every == 0  and self.steps != 0:
                num_batches = math.ceil(CALC_FID_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / sefl.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps}, {fid} \n') 
        self.steps += 1

    @torch.no_grad()
    def evaulate(self, num = 0, num_image_title = 8, trunc = 1.0):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_title


        #latents and noise

        latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

        #regular
        generated_images = self.generate_treancated(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f"{str(num)}.{ext}"), nrow = num_rows)

        #moving average

        generated_images = self.generate_treancated(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f"{str(num)}-ema.{ext}"), nrow = num_rows)
    
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / "fid_real") + "/"
        fake_path = str(self.results_dir / self.name / "fid_fake") + "/"

        #remove any extising files used for fid calculation and recreate directories
        rmtree(real_path, ignore_erros = True)
        rmtree(real_path, ignore_erros = True)
        os.makedirs(real_path)
        os.makedirs(fake_path)


        for batch_norm in tqdm(range(num_batches), desc = "calculating FID - saving reals"):
            real_batch = next(self.loader)
            for k in range(real_batch.size(0)):
                torchvision.utils.save_image(real_batch[k, :, :, :], real_path + "{}.png".format(k + batch_num * self.batch_size))
        
        #generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc = "calculating FID - saving generated"):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            #moving average
            generated_images = self.generate_truncated(self.GAN.GE, latents)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :], str[path(fake_path) / f"{str(j + batch_num * self.batch_size)}-ema{ext}"))
        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)

    @torch.no_grad()
    def generated_truncated(self, G, style, trunc_psi = 0.75, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)
    
    @torch.no_grad()
    def generated_interpolation(self, num = 0, num_image_titles = 8, trunc = 1.0, num_setps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_titles

        latent_dim = self.GAN.latent_dim

        # latents and noise

        latent_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latent_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_setps)

        frames = []

        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_truncated(self.GAN.GE, interp_latents)
            image_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(image_grid.cpu())

            if self.transparent:
                background = Image.new('RGBA', pil_images.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
            
            frames.append(pil_image)
        
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all = True, append_images = frames[1:], duration = 80, loop = 0, optimizer = True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num')
            folder_path.mkdir(parents = True, exist_ok = True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.ext{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists)d[1]]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')
    
    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok = True)
        (self.models_dir / self.name).mkdir(parents = True, exist_ok = True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir/ self.name), True)
        rmtree(str(self.config_path), True)
        self.init_folders()
    
    def save(self, num):
        save_data = {
            "GAN" : self.GAN.state_dict(),
            "version" : __version__, 
            "G_scaler" : self.G_scaler.state_dict(),
            "D_scaler" : self.D_scaler.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num - 1):
        self.load_config()

        name = num

        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            save_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data and self.is_main:
            print(f'loading from version {load_data["version]}')
        try:
            self.GAN.load_state_dict(load_data["GAN"])
        except Exception as e:
            print("unable to laod save model. please try downgrading the version specified by the saved model")
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data["G_scaler"])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data["D_scaler"])

        
    
