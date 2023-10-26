#!/bin/bash
python katacv/G_VAE/g_vae.py --train --wandb-track
python katacv/G_VAE/vae.py --train --wandb-track
