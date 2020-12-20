import torch
from torch.nn import nn
from torch.functional import F



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

