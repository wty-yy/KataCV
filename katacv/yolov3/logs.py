from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'loss_train': MeanMetric(),
        'loss_coord_train': MeanMetric(),
        'loss_conf_train': MeanMetric(),
        'loss_noobj_train': MeanMetric(),
        'loss_class_train': MeanMetric(),
        'mAP_train': MeanMetric(),
        'coco_mAP_train': MeanMetric(),

        'loss_val': MeanMetric(),
        'loss_coord_val': MeanMetric(),
        'loss_conf_val': MeanMetric(),
        'loss_noobj_val': MeanMetric(),
        'loss_class_val': MeanMetric(),
        'mAP_val': MeanMetric(),
        'coco_mAP_val': MeanMetric(),

        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    },
    folder2name={
        'train/metrics': ['loss_train', 'loss_coord_train', 'loss_conf_train', 'loss_noobj_train', 'loss_class_train', 'mAP_train', 'coco_mAP_train'],
        'val/metrics': ['loss_val', 'loss_coord_val', 'loss_conf_val', 'loss_noobj_val', 'loss_class_val', 'mAP_val', 'coco_mAP_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch']
    }
)