# classification
自然场景分类，分别使用vgg16、resnet、senet(res50)、densenet进行尝试
  
  trynet.py负责对模型进行训练及预测
  
  model.py负责设置使用模型及部分超参数
    
  vgg16.py搭建vgg16网络
      
  resnet_with_senet.py搭建resnet网络并可设置是否使用SE
        
  densenet.py搭建desnet169网络
          
  scene_input.py和im2tf.py负责数据读入，分别对应图像格式及TFRecords格式
