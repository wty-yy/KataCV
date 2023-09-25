DATASET_SIZE = {
    'PASCAL': {
        'train': 16551,
        'val': 4952,
        '8examples': 8,
        '100examples': 103,
    },
    'COCO': {
        'train': 117264,  # 15:22/iter only load
        'val': 4954,      # 33 s/iter only load, 142 s/iter with preprocess
        '8examples': 8,   # 33 sec pre iter
    }
}