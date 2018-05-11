# -*- coding: utf-8 -*-

import vgg16 as vgg16
import resnet_with_senet as resnet
import densenet as densenet
import tensorflow as tf

class network(object):  
    def __init__(self,
                 num_classes = 80,
                 model_name = None,
                 scope=None, 
                 dropout_rate=None,            
                 weight_decay = 0.0001,
                 use_batch_norm = True,
                 batch_norm_decay=0.997,
                 batch_norm_epsilon=1e-5,
                 batch_norm_scale=True):
        """
        定义指定网络模型及超参数
        """
        self.n_classes = num_classes 
        self.model = model_name
        self.scope = scope
        self.dropprob = dropout_rate
        self.l2_rate = weight_decay
        self.use_bn = use_batch_norm
        self.bn_decay = batch_norm_decay
        self.bn_epsilon = batch_norm_epsilon
        self.bn_scale = batch_norm_scale
        
        if self.model == None:
            self.model = "densenet"
            
    def inference(self, inputs, is_training = True, reuse = False):
        """
        网络前向传播计算，输出logits张量，keep_prob为drop out参数，预测时置为1
        """
        if self.model == "vgg16":
            if self.scope == None:
                self.scope = 'vgg_16'

            logits = vgg16.vgg_16(inputs = inputs, num_classes = self.n_classes, is_training = is_training,reuse = reuse, 
                                  dropout_keep_prob=self.dropprob,
                                  scope=self.scope,
                                  weight_decay = self.l2_rate,
                                  use_batch_norm = self.use_bn,
                                  batch_norm_decay = self.bn_decay,
                                  batch_norm_epsilon = self.bn_epsilon,
                                  batch_norm_scale = self.bn_scale)
        elif self.model == "res50":
            if self.scope == None:
                self.scope = 'resnet_v1_50'
            logits = resnet.resnet_50(inputs = inputs, num_classes = self.n_classes,is_training = is_training,reuse = reuse,
                                      use_se_module = False,
                                      scope=self.scope,
                                      weight_decay = self.l2_rate,
                                      use_batch_norm = self.use_bn,
                                      batch_norm_decay = self.bn_decay,
                                      batch_norm_epsilon = self.bn_epsilon,
                                      batch_norm_scale = self.bn_scale)
        elif self.model ==  "res50_senet":
            if self.scope == None:
                self.scope = 'resnet_v1_50'
            logits = resnet.resnet_50(inputs = inputs, num_classes = self.n_classes, is_training = is_training,reuse = reuse,
                                      use_se_module = True,
                                      scope=self.scope,
                                      weight_decay = self.l2_rate,
                                      use_batch_norm = self.use_bn,
                                      batch_norm_decay = self.bn_decay,
                                      batch_norm_epsilon = self.bn_epsilon,
                                      batch_norm_scale = self.bn_scale)
        elif self.model == "densenet":
            if self.scope == None:
                self.scope = 'densenet169'
            logits = densenet.densenet169(inputs = inputs, num_classes = self.n_classes, is_training = is_training,reuse = reuse,
                                          dropout_keep_prob=self.dropprob,
                                          scope=self.scope,
                                          weight_decay = self.l2_rate,
                                          use_batch_norm = self.use_bn,
                                          batch_norm_decay = self.bn_decay,
                                          batch_norm_epsilon = self.bn_epsilon,
                                          batch_norm_scale = self.bn_scale)                                
        else:
            raise ValueError("Unknown cost function: "%cost_name)   
        return tf.squeeze(logits)
    