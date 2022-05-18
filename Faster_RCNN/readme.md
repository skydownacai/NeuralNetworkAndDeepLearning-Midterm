# 目标检测实现之Faster R-CNN
## 数据介绍 
该项目使用VOC数据集，此数据集是目标检测中常用数据，其中以2007和2012的数据最为常用，本项目就选用了2007和2012的数据作为训练集，包含共16551张图片及其中目标物体的类别和bounding box位置；测试集为2007年数据中给出的4952张测试图片，两者划分方式是由官方网站指定给出的。
## 文件及代码介绍
img、my_images: 用于存放用来于测试的图片
model_data: 用于训练模型及模型预测类别定义，已上传至百度云:

链接：https://pan.baidu.com/s/151bNif-QgcW7WYON0EsrAA 
提取码：n2pr

nets: faster rcnn网络模型架构python脚本，包含rpg.py和resnet50.py等模块
frcnn.py: 训练、检测代码调用的模型配置文件
get_map.py: 用于获取mAP
voc_annotation.py: 用于自动生成训练VOC数据集的类别文件
train.py: 用于模型的训练
predict.py: 用于用于测试模型在图片中效果
