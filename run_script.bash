#!/bin/bash
# ./run_script.bash 7 32 YOLOv5_b32 0
# ./run_script.bash 6 32 YOLOv5_b32_stopD 0
# ./run_script.bash 5 16 YOLOv5_b16_stopD 7
# CUDA_VISIBLE_DEVICES=$1 python katacv/yolov5/train.py --path-dataset "/data/user/wutianyang/dataset/coco" --path-darknet-weights "/data/user/wutianyang/Coding/models/YOLOv5/NewCSPDarkNet53-0050-lite" --train --wandb-track --batch-size $2 --model-name $3 --load-id $4
CUDA_VISIBLE_DEVICES=$1 python katacv/yolov5/train.py --path-dataset "/data/user/wutianyang/dataset/coco" --path-darknet-weights "" --train --wandb-track --batch-size $2 --model-name $3 --load-id $4
# python katacv/G_VAE/g_vae.py --train --wandb-track --concat-num 1
# python katacv/G_VAE/g_vae.py --train --wandb-track --concat-num 0
# python katacv/G_VAE/vae.py --train --wandb-track
