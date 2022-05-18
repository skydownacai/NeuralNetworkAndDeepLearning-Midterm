# 目标检测实现之YOLOv3
## 数据介绍 
该项目使用VOC数据集，此数据集是目标检测中常用数据，其中以2007和2012的数据最为常用，本项目就选用了2007和2012的数据作为训练集，包含共16551张图片及其中目标物体的类别和bounding box位置；测试集为2007年数据中给出的4952张测试图片，两者划分方式是由官方网站指定给出的。

## 文件及代码介绍

data: 各类数据集，包括VOC等的训练调用yaml脚本；还存放测试用的图片（images、my_images）
export.py: 用于导入模型
models: yolov3模型存放文件，供训练前选定使用的模型类型
runs: 用于存放训练、验证以及测试实验的数据、过程文件以及结果图片等
setup.cfg: 用于解析构建darknet网络架构 
yolov3.pt: 用于保存训练后的模型
train.py: 用于模型训练
detect.py: 用于测试模型在图片中效果



模型与部分文件上传至网盘

链接：https://pan.baidu.com/s/1jOVZFOrShPFUmXerNKaKcg 
提取码：jbbb

下载覆盖到根目录即可
