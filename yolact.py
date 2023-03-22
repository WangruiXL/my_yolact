import torch
from torch import nn as nn
from data.config import cfg, mask_type

# 什么是JIT
'''首先要知道 JIT 是一种概念，全称是 Just In Time Compilation，中文译为「即时编译」，是一种程序优化的方法，
一种常见的使用场景是「正则表达式」
# 如果多次使用到某一个正则表达式，则建议先对其进行 compile，然后再通过 compile 之后得到的对象来做正则匹配。
# 而这个 compile 的过程，就可以理解为 JIT（即时编译）
'''
# 当可用的GPU超过一个的时候，才会使用JIT优化程序
use_jit = torch.cuda.device_count() <= 1  # return True or False
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')


class Yolact(nn.Module):
