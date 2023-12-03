from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'loss_noobj_train': MeanMetric(),
    'loss_coord_train': MeanMetric(),
    'loss_obj_train': MeanMetric(),
    'loss_class_train': MeanMetric(),

    'P@50_val': MeanMetric(),
    'R@50_val': MeanMetric(),
    'AP@50_val': MeanMetric(),
    'AP@75_val': MeanMetric(),
    'mAP_val': MeanMetric(),
    
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'metrics/train': [
      'loss_train',
      'loss_noobj_train',
      'loss_coord_train',
      'loss_obj_train',
      'loss_class_train',
    ],
    'metrics/val': [
      'P@50_val',
      'R@50_val',
      'AP@50_val',
      'AP@75_val',
      'mAP_val',
    ],
    'charts': [
      'SPS', 'SPS_avg',
      'epoch', 'learning_rate'
    ]
  }
)