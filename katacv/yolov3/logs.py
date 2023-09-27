from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'cost_train': MeanMetric(),
        'loss_train': MeanMetric(),
        'regular_train': MeanMetric(),
        'loss_noobj_train': MeanMetric(),
        'loss_coord_train': MeanMetric(),
        'loss_obj_train': MeanMetric(),
        'loss_class_train': MeanMetric(),

        'loss_val': MeanMetric(),
        'loss_noobj_val': MeanMetric(),
        'loss_coord_val': MeanMetric(),
        'loss_obj_val': MeanMetric(),
        'loss_class_val': MeanMetric(),

        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric(),
        'learning_rate': 0,
    },
    folder2name={
        'metrics/train': [
            'cost_train', 'regular_train',
            'loss_train', 
            'loss_noobj_train',
            'loss_coord_train',
            'loss_obj_train',
            'loss_class_train',
        ],
        'metrics/val': [
            'loss_val',
            'loss_noobj_val',
            'loss_coord_val',
            'loss_obj_val',
            'loss_class_val',
        ],
        'charts': ['SPS', 'SPS_avg', 'epoch', 'learning_rate']
    }
)