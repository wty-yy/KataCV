from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'loss_val': MeanMetric(),
    # 'AP50_val': MeanMetric(),
    # 'AP75_val': MeanMetric(),
    # 'AP_val': MeanMetric(),
    
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'metrics': [
      'loss_train',
      'loss_val',
      # 'AP50_val',
      # 'AP75_val',
      # 'AP_val'
    ],
    'charts': [
      'SPS', 'SPS_avg',
      'epoch', 'learning_rate'
    ]
  }
)