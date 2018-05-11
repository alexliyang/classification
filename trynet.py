# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import scene_input
import numpy as np
import time
import argparse
import cv2
import json
import os
import img2tf
import model 

BATCH_SIZE = 32
IMAGE_SIZE = 264
CROP_SIZE = 224
IMAGE_CHANNEL = 3
NUM_CLASS = 80
CHECKFILE = './checkpoint/'#
LOGNAME = 'scene'

learning_rate = 0.001
optimizer = 'sgd'
iters = 5000
model_name = 'densenet'
restore_exclusions = ['densenet169/logits']#'vgg_16/fc8' 'resnet_v1_50/logits'
monitor_layers=['densenet169/logits','densenet169/Conv_9','densenet169/Conv_8'] #'vgg_16/fc8','resnet_v1_50/logits','resnet_v1_50/block4'
validate_size = 20  #乘以Batcn Size
use_bn = True

filewriter_path = "./tensorboard/"

test_dir = '/you/path/for/validation/data'
test_annotations = '/you/path/for/validation/data/annotations'

def label_smoothing(label, smooth_rate,num_class = NUM_CLASS):
    label = tf.convert_to_tensor(label)
    
    alpha = tf.scalar_mul(smooth_rate, tf.ones_like(label, dtype=tf.float32))
    label = tf.where(tf.equal(label, 1.0), alpha, (1-alpha)/(num_class-1))
    return label

def focal_loss(logits, onehot_labels, alpha=0.125, gamma=2):

    logits = tf.convert_to_tensor(logits)
    onehot_labels = tf.convert_to_tensor(onehot_labels)

    precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
    onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
    predictions = tf.nn.sigmoid(precise_logits)
    predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
    epsilon = 1e-8
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                  axis=1)
    return losses

def get_cost(logits, one_hot_labels, cost_name = "cross_entropy", regularizer = True, smooth_rate = None, opt_kwargs = {}):
    """
    计算loss   
    """
        
    if cost_name == "focal_loss":
        '读入focal loss 参数'
        alpha = opt_kwargs.pop("alpha", 0.125)
        gamma = opt_kwargs.pop("gamma", 2)
        loss = tf.reduce_sum(focal_loss(logits, one_hot_labels,alpha, gamma))
            
    else:
        '判断是否使用label smoothing'
        if smooth_rate is not None:
            one_hot_labels = label_smoothing(one_hot_labels,smooth_rate,NUM_CLASS)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    
    'L2正则化'
    if regularizer:
        loss = loss  + tf.reduce_sum(tf.losses.get_regularization_losses())      
    return loss


def get_optimizer(loss, training_iters, global_step, optimizer ="sgd", opt_kwargs={}):
    """
    设置优化方法及参数
    """
    #global_step = tf.Variable(0)
    if optimizer == "momentum":
        learning_rate = opt_kwargs.pop("learning_rate", 0.1)
        decay_rate = opt_kwargs.pop("decay_rate", 0.95)
        momentum = opt_kwargs.pop("momentum", 0.2)
        
        learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                    global_step=global_step, 
                                                    decay_steps=training_iters,  
                                                    decay_rate=decay_rate, 
                                                    staircase=True)
        
        opti = tf.train.MomentumOptimizer(learning_rate=learning_rate_node, momentum=momentum,
                                               **opt_kwargs).minimize(loss, 
                                                                            global_step=global_step)
    elif optimizer == "adam":
        learning_rate = opt_kwargs.pop("learning_rate", 0.001)
        learning_rate_node = tf.Variable(learning_rate)
        
        opti = tf.train.AdamOptimizer(learning_rate=learning_rate_node, 
                                           **opt_kwargs).minimize(loss,
                                                                 global_step=global_step)
    elif optimizer == "sgd":
        learning_rate = opt_kwargs.pop("learning_rate", 0.1)
        decay_rate = opt_kwargs.pop("decay_rate", 0.95)
        
        learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                    global_step=global_step, 
                                                    decay_steps=training_iters,  
                                                    decay_rate=decay_rate, 
                                                    staircase=True)
        
        opti = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_node).minimize(loss,
                                                                 global_step=global_step)
    
    return opti,  learning_rate_node   

def get_restore_var(exclusions = []):
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore
                    
def train_tf(train_dir, max_step, checkpoint_dir=CHECKFILE):
        # train the model
    img,lab = img2tf.decode_from_tfrecords(train_dir,image_size=IMAGE_SIZE)
    features,labels = img2tf.get_batch(img,lab,batch_size=BATCH_SIZE,crop_size=CROP_SIZE)  
    is_training = tf.placeholder("bool")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASS)
    
    
    net = model.network(NUM_CLASS, model_name, use_batch_norm = use_bn)
    logits = net.inference(inputs = features, is_training= is_training,reuse = False)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    loss = get_cost(logits, one_hot_labels, regularizer=False)
    
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    opti, learning_rate_node = get_optimizer(loss = loss, training_iters = iters, global_step = global_step, 
                                             optimizer =optimizer,  opt_kwargs={'learning_rate':learning_rate})

    
    
    var_list = []
    for scope in monitor_layers:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        var_list.extend(variables)
    for var in var_list:
        tf.summary.histogram(var.name, var)    
    tf.summary.histogram('logdits', logits)
    tf.summary.scalar('learning rate', learning_rate_node)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filewriter_path)  
    
    testfeatures = tf.placeholder("float32", shape=[None, CROP_SIZE, CROP_SIZE, IMAGE_CHANNEL], name="testfeatures")
    testlabels = tf.placeholder("float32", [None], name="labels")
    test_data = scene_input.scene_data_fn(test_dir, test_annotations)  
    test_one_hot_labels = tf.one_hot(indices=tf.cast(testlabels, tf.int32), depth=80)
    testlogits = net.inference(inputs = testfeatures, is_training= is_training,reuse = True)
    test_correct_prediction = tf.equal(tf.argmax(testlogits, 1), tf.argmax(test_one_hot_labels, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)  
          
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = get_restore_var(exclusions = restore_exclusions)
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            if str.isdigit():
                start_step = int(str)
            else:
                start_step = 0
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
       
         
        logger = scene_input.train_log(LOGNAME)

        for step in range(start_step, start_step + max_step):
            start_time = time.time()
            sess.run([features,one_hot_labels,loss, opti],feed_dict={is_training: True})
            if step % 200 == 0:
                train_accuracy_acc=0
                train_loss_acc=0
                total_loss_acc=0
                for count in range(validate_size):                    
                    train_accuracy,total_loss = sess.run([accuracy,loss], feed_dict={ is_training: False})
                    train_accuracy_acc+=train_accuracy
                    total_loss_acc+=total_loss
                train_accuracy_acc=train_accuracy_acc/20
                total_loss_acc=total_loss_acc/20                
                duration = time.time() - start_time
                logger.info("step %d: training accuracy %g, loss: %g ,(%0.3f sec)" % (step, train_accuracy_acc,total_loss_acc, duration))
                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy_acc), 
                    ])
                writer.add_summary(summary, step)
                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="total_loss", simple_value=total_loss_acc), 
                    ])
                writer.add_summary(summary, step)
                sums = sess.run(merged_summary, feed_dict={ is_training: False})
                writer.add_summary(sums, step)
                
            if step % 1000 == 0:
                acctotal=0.0
                for count in range(validate_size):
                    test_x, test_y = test_data.next_batch(BATCH_SIZE, CROP_SIZE)
                    test_acc = sess.run(test_accuracy, feed_dict={testfeatures: test_x, testlabels: test_y,is_training:False})
                    acctotal = test_acc+acctotal
                acctotal = acctotal / 20 
                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="test_acc", simple_value=acctotal), 
                    ])
                writer.add_summary(summary, step)              
                logger.info("step %d: test accuracy %g *******" % (step, acctotal))                
                saver.save(sess, CHECKFILE+'model.ckpt', global_step=step)
                print('writing checkpoint at step %s' % step)
        
        coord.request_stop()
        coord.join(threads) 


def test(test_dir, checkpoint_dir='./checkpoint_res50/'):
    # predict the result 
    test_images = os.listdir(test_dir)
    features = tf.placeholder("float32", shape=[None, CROP_SIZE, CROP_SIZE, IMAGE_CHANNEL], name="features")
    labels = tf.placeholder("float32", [None], name="labels")
    is_training = tf.placeholder("bool")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=80)
    
    net = model.network(NUM_CLASS, model_name, use_batch_norm = use_bn)
    logits = net.inference(inputs = features, is_training= is_training,reuse = False)
    values, indices = tf.nn.top_k(logits, 3)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')

        result = []
        for test_image in test_images:
            temp_dict = {}
            x = scene_input.img_resize(os.path.join(test_dir, test_image), CROP_SIZE)
            indices_eval = sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), is_training: False})
            predictions = np.squeeze(indices_eval)
            temp_dict['image_id'] = test_image
            temp_dict['label_id'] = predictions.tolist()
            result.append(temp_dict)
            print('image %s is %d,%d,%d' % (test_image, predictions[0], predictions[1], predictions[2]))
        
        with open('submit.json', 'w') as f:
            json.dump(result, f)
            print('write result json, num is %d' % len(result))  
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help="""\
        determine train or test\
        """
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='/your/path/for/train/data/',
        help="""\
        determine path of trian images\
        """
    )

    parser.add_argument(
        '--annotations',
        type=str,
        default='/your/path/for/annotations/',
        help="""\
        annotations for train images\
        """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/your/path/for/test/data/',
        help="""\
        determine path of test images\
        """
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=30001,
        help="""\
        determine maximum training step\
        """
    )

    parser.add_argument(
        '--model_dir',
        type=int,
        default='/your/path/for/model/dir/',
        help="""\
        determine maximum training epochs\
        """
    )

    FLAGS = parser.parse_args()
    #FLAGS.mode = 'test'

    if FLAGS.mode == 'train':
        #train(FLAGS.train_dir, FLAGS. annotations, FLAGS.max_step)
        train_tf([FLAGS.train_dir], FLAGS.max_step,FLAGS.model_dir)

    elif FLAGS.mode == 'test':
        test(FLAGS.test_dir)
    else:
        raise Exception('error mode')
    print('done')
