import torch
from backbone import ResNetBackbone
COCO_CLASSES = ('trepang', 'starfish', 'shell', 'sea urchin',)

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4}


class Config:
    def __init__(self, config_dict):
        for key, val in config_dict:
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        # 现在的self不是字典，是(key,val),需要内置函数vars():argument->object.__dict__.
        ret = Config(vars(self))
        # Without arguments, equivalent to locals().返回一个字典
        # With an argument, equivalent to object.__dict__.接受的argument可以是类、模块、实例

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret
        # 把self copy 到ret中

    def replace(self, new_config_dict={}):
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)
        for key, val in new_config_dict.item():
            self.__setattr__(key, val)
        # 把new_config_dict中的内容更新到self中

    def print(self):
        for k, v in vars(self).items():
            print(k, '=', v)


# 数据库
dataset_base = Config(
    {'name': 'Base Dataset',
     # Training images and annotations
     'train_images': './data/coco/train2017/',
     'train_info': './data/coco/annotations/instances_train2017.json',

     # Validation images and annotations.
     'valid_images': './data/coco/val2017/',
     'valid_info': './data/coco/annotations/instances_val2017.json',
     # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
     'has_gt': True,

     # A list of names for each of you classes.
     'class_names': COCO_CLASSES,

     # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
     # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
     # If not specified, this just assumes category ids start at 1 and increase sequentially.
     'label_map': None

     })
coco2014_dataset = dataset_base.copy(
    {
        'name': 'COCO 2014',

        'train_info': './data/coco/annotations/instances_train2014.json',
        'valid_info': './data/coco/annotations/instances_val2014).json',

        'label_map': COCO_LABEL_MAP
    })
coco2017_dataset = dataset_base.copy(
    {
        'name': 'COCO 2017',

        'train_info': './data/coco/annotations/instances_train2017.json',
        'valid_info': './data/coco/annotations/instances_val2017.json',

        'label_map': COCO_LABEL_MAP
    }

)
# ----------------------- MASK BRANCH TYPES ----------------------- #
'''
使用全卷积层(fc layers)生成语义向量，使用卷积层(conv layers)生成掩模原型***************************
'''
mask_type = Config(
    {
        # we develop an“fc-mask” model that produces masks for each anchor \
        # as the reshaped output of an fc layer
        # Direct produces masks directly as the output of each pred module.
        # Parameters: mask_size, use_gt_bboxes 掩膜尺寸，使用梯度边界框
        'direct': 0,
        # Lincomb produces coefficients as the output of each pred module then uses those coefficients
        # to linearly combine features from a prototype network to create image-sized masks.
        'lincomb': 1,
        # Parameters:
        #   - masks_to_train (int): Since we're producing (near) full image masks, it'd take too much
        #                           vram to backprop on every single mask. Thus we select only a subset.
        #   - mask_proto_src (int): The input layer to the mask prototype generation network. 掩膜原型生成网络的输入层
        #                           This is an index in backbone.layers. Use to use the image itself instead.
        #   - mask_proto_net (list<tuple>): A list of layers in the mask proto network with the last one
        #                                   being where the masks are taken from. Each conv layer is in
        #                                   the form (num_features, kernel_size, **kwdargs). An empty
        #                                   list means to use the source for prototype masks. If the
        #                                   kernel_size is negative, this creates a deconv layer instead.
        #                                   If the kernel_size is negative and the num_features is None,
        #                                   this creates a simple bilinear interpolation layer instead.**
        #   - mask_proto_bias (bool): Whether to include an extra coefficient that corresponds to a proto
        #                             mask of all ones.
        #   - mask_proto_prototype_activation (func): The activation to apply to each prototype mask.
        #   - mask_proto_mask_activation (func): After summing the prototype masks with the predicted
        #                                        coeffs, what activation to apply to the final mask.
        #   - mask_proto_coeff_activation (func): The activation to apply to the mask coefficients.
        #   - mask_proto_crop (bool): If True, crop the mask with the predicted bbox during training.
        #   - mask_proto_crop_expand (float): If cropping, the percent to expand the cropping bbox by
        #                                     in each direction. This is to make the model less reliant
        #                                     on perfect bbox predictions.
        #   - mask_proto_loss (str [l1|disj]): If not None, apply an l1 or disjunctive regularization
        #                                      loss directly to the prototype masks.
        #   - mask_proto_binarize_downsampled_gt (bool): Binarize GT after dowsnampling during training?
        #   - mask_proto_normalize_mask_loss_by_sqrt_area (bool): Whether to normalize mask loss by sqrt(sum(gt))
        #   - mask_proto_reweight_mask_loss (bool): Reweight mask loss such that background is divided by
        #                                           #background and foreground is divided by #foreground.
        #   - mask_proto_grid_file (str): The path to the grid file to use with the next option.
        #                                 This should be a numpy.dump file with shape [numgrids, h, w]
        #                                 where h and w are w.r.t. the mask_proto_src convout.
        #   - mask_proto_use_grid (bool): Whether to add extra grid features to the proto_net input.
        #   - mask_proto_coeff_gate (bool): Add an extra set of sigmoided coefficients that is multiplied
        #                                   into the predicted coefficients in order to "gate" them.
        #   - mask_proto_prototypes_as_features (bool): For each prediction module, downsample the prototypes
        #                                 to the convout size of that module and supply the prototypes as input
        #                                 in addition to the already supplied backbone features.
        #   - mask_proto_prototypes_as_features_no_grad (bool): If the above is set, don't backprop gradients to
        #                                 to the prototypes from the network head.
        #   - mask_proto_remove_empty_masks (bool): Remove masks that are downsampled to 0 during loss calculations.
        #   - mask_proto_reweight_coeff (float): The coefficient to multiple the forground pixels with if reweighting.
        #   - mask_proto_coeff_diversity_loss (bool): Apply coefficient diversity loss on the coefficients so that the same
        #                                             instance has similar coefficients.
        #   - mask_proto_coeff_diversity_alpha (float): The weight to use for the coefficient diversity loss.
        #   - mask_proto_normalize_emulate_roi_pooling (bool): Normalize the mask loss to emulate roi pooling's affect on loss.
        #   - mask_proto_double_loss (bool): Whether to use the old loss in addition to any special new losses.
        #   - mask_proto_double_loss_alpha (float): The alpha to weight the above loss.
        #   - mask_proto_split_prototypes_by_head (bool): If true, this will give each prediction head its own prototypes.
        #   - mask_proto_crop_with_pred_box (bool): Whether to crop with the predicted box or the gt box.

    }
)
# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config(
    {
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'relu': lambda x: torch.nn.functional.relu(x, inplace=True),
        # torch.Relu 作为一个单独的层，nn.functional.relu 是一个函数，用在def forward(x)中
        # inplace=True，把y=x+1,x=y 覆盖为x=x+1 减少一个y的存储，节省运算内存
        'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
        # dim=A dimension along which softmax will be computed.dim=0,1，2，-1
        'none': lambda x: x,

    }
)
# ----------------------- CONFIG DEFAULTS ----------------------- #

coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': 5,  # This should include the background class

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 0.4 / 256 * 140 * 140,  # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,  # 非极大值抑制阈值

    # See mask_type for details.
    'mask_type': mask_type.direct,  # **这里的调用字典的关键字，得到的就是相应的值
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,  # 掩膜生成的网络输入层
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    # 掩码原型网络中的层列表，最后一层是掩码的来源。每个conv层的形式为（num_features、kernel_size、kwdargs）。
    # 空列表表示使用原型掩码的源source。
    # 如果kernel_size为负值，则会创建一个去卷积层。如果kernel_size为负值，num_features为None，则会创建一个简单的双线性插值层。
    'mask_proto_bias': False,
    # 是否包含一个适用于所有原型掩码的偏置系数
    'mask_proto_prototype_activation': activation_func.relu,
    # 每个原型掩膜的激活（实际上就是一些激活函数的应用，这里原型掩膜的激活使用relu函数）
    'mask_proto_mask_activation': activation_func.sigmoid,
    # 在将原型掩模与预测系数相加后，使用sigmoid函数激活最终掩模
    'mask_proto_coeff_activation': activation_func.tanh,
    # 掩膜系数的激活函数
    'mask_proto_crop': True,
    # 如果为True，则在训练过程中使用预测的bbox裁剪掩膜
    'mask_proto_crop_expand': 0,
    # 如果进行裁剪，则将裁剪框在每个方向上（扩展的百分比）。这是为了减少模型的依赖性
    'mask_proto_loss': None,
    # 如果不是None，则将l1或分割的正则化（disjunctive regularization）损失直接应用于原型掩码。
    'mask_proto_binarize_downsampled_gt': True,
    # 在训练过程中进行向下采样后对真实背景二值化（0，1：背景，前景）？
    # Binarize GT after dowsnampling during training?
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    # 是否通过sqrt（sum（gt），平方和）标准化掩码损失
    'mask_proto_reweight_mask_loss': False,
    # 重新加权掩模损失，使得背景被背景分割，前景被前景分割。
    'mask_proto_grid_file': 'data/grid.npy',
    # 要与下一个选项一起使用的梯度文件的路径。这应该是一个numpy.dump文件，其shape为[numgrids，h，w]
    'mask_proto_use_grid': False,
    # 是否向proto_net输入添加额外的梯度特征
    'mask_proto_coeff_gate': False,
    # 添加一组额外的sigmoid过的系数，将其相乘到预测系数中，以便对其进行“gate”。
    'mask_proto_prototypes_as_features': False,
    # 对于每个预测模块，将原型下采样到该模块的卷积大小，并将原型作为输入提供给已经提供的主干特征。
    'mask_proto_prototypes_as_features_no_grad': False,
    # 如果设置了以上内容，则不要从网络头向原型反向投影梯度。
    'mask_proto_remove_empty_masks': False,
    # 删除在损失计算过程中向下采样为0的掩码。
    'mask_proto_reweight_coeff': 1,
    # The coefficient to multiple the forground pixels with if reweighting.
    'mask_proto_coeff_diversity_loss': False,
    # Apply coefficient diversity loss on the coefficients so that the same instance has similar coefficients.
    'mask_proto_coeff_diversity_alpha': 1,
    # The weight to use for the coefficient diversity loss.
    'mask_proto_normalize_emulate_roi_pooling': False,
    # Normalize the mask loss to emulate roi pooling's affect on loss.
    'mask_proto_double_loss': False,
    # Whether to use the old loss in addition to any special new losses.
    'mask_proto_double_loss_alpha': 1,
    # The alpha to weight the above loss. 加权
    'mask_proto_split_prototypes_by_head': False,
    # If true, this will give each prediction head 预测头部 its own prototypes.
    'mask_proto_crop_with_pred_box': False,
    # Whether to crop with the predicted box or the gt box（真实边界框）.

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 550,
    'discard_box_height': 4 / 550,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
    # Note: at the moment this is only implemented if use_focal_loss is on.
    'use_objectness_score': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Adds another branch to the netwok to predict Mask IoU.
    'use_mask_scoring': False,
    'mask_scoring_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size.
    'max_size': 300,

    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,

    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # Fast Mask Re-scoring Network
    # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
    # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
    # then use global pooling to get the final mask score
    'use_maskiou': False,

    # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
    'maskiou_net': [],

    # Discard predicted masks whose area is less than this
    'discard_mask_area': -1,

    'maskiou_alpha': 1.0,
    'rescore_mask': False,
    'rescore_bbox': False,
    'maskious_to_train': -1, })

# ----------------------- TRANSFORMS ----------------------- #
resnet_transform = Config(
    {
        'channel_order': 'RGB',
        'normalize': True,
        'subtract_means': False,
        'to_float': False,
    }
)

# ----------------------- BACKBONES ----------------------- #
backbone_base = Config(
    {
        'name': 'Base Backbone',
        'path': 'path/to/pretrained/weights',
        'type': object,
        'args': tuple(),
        'transform': resnet_transform,

        'selected_layers': list(),
        'pred_scales': list(),
        'pred_aspect_ratios': list(),

        'use_pixel_scales': False,
        'preapply_sqrt': True,
        'use_square_anchors': False,
    }
)

resnet101_backbone = backbone_base.copy(
    {
        'name': 'ResNet101',
        'path': 'resnet101_reducedfc.pth',
        'type': ResNetBackbone,
        'args': ([3, 4, 23, 3],),
        'transform': resnet_transform,

        'selected_layers': list(range(2, 8)),
        'pred_scales': [[1]] * 6,
        'pred_aspect_ratios': [[[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]] * 6,
    }
)

# ----------------------- YOLACT v1.0 CONFIGS ----------------------- #


yolact_base_config = coco_base_config.copy(
    {
        'name': 'yolact_base',
        # Dataset stuff
        'dataset': coco2017_dataset,
        'num_classes': len(coco2017_dataset.class_names) + 1,
        # Image Size
        'max_size': 550,

        # Training params
        'lr_steps': (280000, 600000, 700000, 750000),
        'max_iter': 800000,

        # Backbone Settings
        'backbone': resnet101_backbone.copy()
    }
)
# Default config
cfg = yolact_base_config.copy()


def set_cfg(config_name:str):
    global cfg

    cfg.replace(eval(config_name))
    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name: str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)




