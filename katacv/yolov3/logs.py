from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'cost_train': MeanMetric(),
        'loss_train': MeanMetric(),
        'regular_train': MeanMetric(),

        'loss_val': MeanMetric(),

        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    },
    folder2name={
        'metrics': ['loss_train', 'loss_val', 'cost_train', 'regular_train'],
        'charts': ['SPS', 'SPS_avg', 'epoch']
    }
)