import argparse  # 命令行分析库:将相关参数进行设置
from data import *
import torch.nn as nn
from yolact import Yolact

# 字符串转换为布尔值
#转换原理：if v.lower()字符小写is one of ('yes','true','t','1'),return True. else return False
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


# 实例化ArgumentParser命令行参数解析器
parser = argparse.ArgumentParser(description='my_yolact traing script')
parser.add_argument('--batch_size',
                    default='8',
                    type=int,
                    help='Batch Size of Traing')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.(可以在中断的训练基础上继续训练)')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')  # 不明白为什么是-1
parser.add_argument('--num_workers',
                    default='4',
                    type=int,
                    help='Number of workers used in dataloading(加载数据集的进程)')
parser.add_argument('--cuda', default=True, type=str2bool,  # 为什么要用这个
                    help='Use CUDA to train model')

# 一些超参数
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.(在config设置)')
parser.add_argument('--momentum', default=None, type=float,  # 动量
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,  # 权重衰减
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_floder',
                    default='Weights/',
                    type=str,
                    help='Directory for saving checkpoint models.')
parser.add_argument('--save_log',
                    default='Logs/',
                    type=str,
                    help='Directory for saving logs.')
parser.add_argument('--config',
                    default=None,
                    type=str,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
# action=‘store_true’，只要运行时该变量有传参就将该变量设为True。
parser.add_argument('--keep_latest',
                    dest='keep_latest',
                    action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
# 相当于default的更新
parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
# 解析参数,后可调用命名的参数
args = parser.parse_args()

# 自定义配置，如果这个配置存在的话，就将该配置加入cfg中，默认yolact_base_config(一般训练就用的这个配置)
if args.config is not None:
    set_cfg(args.config)  # from data import* ->from config import* ->def set_cfg

# 自定义数据集，默认yolact_base_config中的数据集是coco2017_dataset,如果args设置了，就用设置的覆盖默认的
if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))
    # 提示调整尺寸?  尺寸缩放比 批量是8的时候factor=1

    cfg.lr *= factor  # 根据缩放比调整学习率以及最大的迭代次数,以及学习率步数
    cfg.max_iter //= factor  # (int)
    # 'lr_steps': (280000, 360000, 400000),
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None:  # 如果args.name=None,则设置args.name=cfg.name,生成器中默认都是None
        # getattr(x, 'y') is equivalent to x.y.
        setattr(args, name, getattr(cfg, name))
    # setattr(x, 'y', v) is equivalent to ``x.y = v''


# 需要更新的参数
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr ##还没看到在哪里管理
cur_lr = args.lr
#   调用exit函数，终止Python程序。exit(num)可省略的参数。通常情况下0表示程序正常退出，
#   1表示程序遇到了某个错误而导致退出。实际运用中可以使用任何整型数据，表示不同的自定义错误类型。
if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

# if __name__=="__main__"所在模块是被直接运行的，则该语句下代码块被运行，
# 如果所在模块是被导入到其他的python脚本中运行的，则该语句下代码块不被运行
if args.batch_size // torch.cuda.device_count() < 6:  # 每个GPU的批量小于6，无法批量标准化，冻结每个层的bn
    if __name__ == "__main__":
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

# 设置默认的tensor类型，如果有cuda，并且设置使用cuda，则设置tensor为cuda的Tensor，所有的计算都在GPU上进行
if torch.cuda.is_available():
    if cfg.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_tensor_type(t)->t (type or string):
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses
