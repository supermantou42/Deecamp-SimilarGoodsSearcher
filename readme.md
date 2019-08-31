# Deecamp 2019 项目实践后端部分
使用Flask搭建的后台

主要逻辑是先用yolo框选出物品，再用网络进行相似搜索

尝试过的网络有ProtoNet，Densenet， TripleLossNet

## Requirements
+ Python 3
+ PyTorch
+ torchvision
+ Tensorflow
+ numpy
+ Pillow
+ opencv
+ Flask

## Usage

选择合适的网络的入口后直接运行，开启Flask后使用Postman测试接口，接口集中在入口文件中，`app-PTNET.py` 中的最为完整