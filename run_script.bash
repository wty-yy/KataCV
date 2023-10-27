#!/bin/bash
python katacv/G_VAE/g_vae.py --train --wandb-track --concat-num 2
python katacv/G_VAE/g_vae.py --train --wandb-track --concat-num 1
python katacv/G_VAE/g_vae.py --train --wandb-track --concat-num 0
# python katacv/G_VAE/vae.py --train --wandb-track
