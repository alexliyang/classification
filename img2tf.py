# -*-  coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import json
import cv2
import logging
import PIL.Image as Image
from matplotlib import pyplot as plt
nchannel = 3

def img_resize(imgpath, img_size):
    img = Image.open(imgpath)
    width = img.width
    height = img.height
    if (width > height):
        scale = float(img_size) / float(width)
        img = np.array(cv2.resize(np.array(img), (
        img_size, int(height * scale)))).astype(np.float32)
    else:
        scale = float(img_size) / float(height)
        img = np.array(cv2.resize(np.array(img), (
        int(width * scale), img_size))).astype(np.float32)
    
    width = img.shape[1]
    height = img.shape[0]
    padx = (img_size-height) //2
    pady = (img_size-width) //2
    img = np.array(img).astype(np.float32)*1.0/255-0.5
    res = np.array(cv2.copyMakeBorder(img, padx, img_size-height-padx, pady, img_size-width-pady, cv2.BORDER_CONSTANT, 
            value = [0,0,0])).astype(np.float32)
    return res
    
def encode_to_tfrecords(lable_file,data_root,output_path, image_size = 350):  
    writer=tf.python_io.TFRecordWriter(output_path)  
    num_example=0  
    with open(lable_file, 'r') as f:
        label_list = json.load(f)
    for image in label_list:
        label = int(image['label_id'])
        image = img_resize(os.path.join(data_root, image['image_id']),image_size)
  
        example=tf.train.Example(features=tf.train.Features(feature={   
            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),  
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  
            }))  
        serialized=example.SerializeToString()  
        writer.write(serialized) 
        num_example+=1
        print num_example         
    print lable_file,"样本数据量：",num_example  
    writer.close()  
    
 #读取tfrecords文件  
def decode_from_tfrecords(filename,image_size=350,nchannel = nchannel, num_epoch=None):    
    filename_queue=tf.train.string_input_producer(filename,num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数  
    reader=tf.TFRecordReader()  
    _,serialized=reader.read(filename_queue)  
    example=tf.parse_single_example(
        serialized,
        features={   
            'image':tf.FixedLenFeature([],tf.string),  
            'label':tf.FixedLenFeature([],tf.int64)  
        })
    
    image=tf.decode_raw(example['image'],tf.float32) 
    label=tf.cast(example['label'], tf.int32) 
   
    image = tf.reshape(image,[image_size,image_size,nchannel])
  
    return image,label 
  
def get_batch(image, label,batch_size,crop_size):  
    
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪  
    distorted_image = tf.image.random_flip_left_right(distorted_image)#左右随机翻转 
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=0.24)#亮度变化  
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,num_threads=10,capacity=1000,min_after_dequeue=32 ) 

    return images, label_batch   

#这个是用于测试阶段，使用的get_batch函数  
def get_test_batch(image, label, batch_size):  
    images, label_batch=tf.train.batch([image, label],batch_size=batch_size)  
    return images, label_batch   
  
#测试上面的压缩、解压代码  
def testread(path):  
    image,label=decode_from_tfrecords(path)  
    images, sparse_labels=get_batch(image, label, 1)
    init=tf.global_variables_initializer()
    with tf.Session() as session:  
        session.run(init)  
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=session)  
        for l in range(10):
            images_np,batch_label_np=session.run([images,sparse_labels])  
            print images_np.shape  
            print batch_label_np.shape
            img=np.ushort((np.reshape(images_np, (224,224,3))*255+127))
            print img.shape
            plt.figure()
            plt.imshow(img)
            plt.show()
                     
        coord.request_stop()
        coord.join(threads)