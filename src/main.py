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

            