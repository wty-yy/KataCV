#!/bin/bash
python katacv/G_VAE/g_vae.py --train --wandb-track --feature-size 1024
python katacv/G_VAE/g_vae.py --train --wandb-track --feature-size 2048
python katacv/G_VAE/g_vae.py --train --wandb-track --feature-size 4096
python katacv/G_VAE/vae.py --train --wandb-track --feature-size 1024
python katacv/G_VAE/vae.py --train --wandb-track --feature-size 2048
python katacv/G_VAE/vae.py --train --wandb-track --feature-size 4096
