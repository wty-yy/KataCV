from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'loss_train': MeanMetric(),
        'acc_train': MeanMetric(),
        'loss_val': MeanMetric(),
        'acc_val': MeanMetric(),

        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric(),
        'epoch': 0,
        'learning_rate': 0,
    },
    folder2name={
        'metrics': [
            'loss_train',
            'acc_train',
            'loss_val',
            'acc_val',
        ],
        'charts': [
            'SPS',
            'SPS_avg',
            'epoch',
            'learning_rate',
        ]
    }
)