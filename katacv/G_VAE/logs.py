from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'loss_img_train': MeanMetric(),
    'loss_kl_train': MeanMetric(),
    'loss_cls_train': MeanMetric(),
    'acc_train': MeanMetric(),

    'loss_val': MeanMetric(),
    'loss_img_val': MeanMetric(),
    'loss_kl_val': MeanMetric(),
    'loss_cls_val': MeanMetric(),
    'acc_val': MeanMetric(),

    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'epoch': 0,
  },
  folder2name={
    'metrics/train': [
      'loss_train',
      'loss_img_train',
      'loss_kl_train',
      'loss_cls_train',
      'acc_train',
    ],
    'metrics/val': [
      'loss_val',
      'loss_img_val',
      'loss_kl_val',
      'loss_cls_val',
      'acc_val',
    ],
    'charts': [
      'SPS',
      'SPS_avg',
      'epoch',
    ]
  }
)