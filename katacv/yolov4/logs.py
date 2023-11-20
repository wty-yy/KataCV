from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'loss_val': MeanMetric(),
    
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'metrics': [
      'loss_train',
      'loss_val'
    ],
    'charts': [
      'SPS', 'SPS_avg',
      'epoch', 'learning_rate'
    ]
  }
)