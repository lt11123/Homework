

# PyTorch实现CIFAR10

利用 PyTorch 在 CIFAR10 数据集上实现深度神经网络

### 1、查看数据（格式，大小，形状）

**（1）数据格式**

```
trainset.class_to_idx
```

{'airplane': 0,
 'automobile': 1,
 'bird': 2,
 'cat': 3,
 'deer': 4,
 'dog': 5,
 'frog': 6,
 'horse': 7,
 'ship': 8,
 'truck': 9}

**（2）数据大小与形状**

```
trainset.data.shape
```

(50000, 32, 32, 3)

**（3）显示数据集中的部分图片及标签**

![image-20211108193252778](D:\研\人工智能安全\homework\images\image-20211108193252778.png)

### 2、定义网络模型

**（1）模型结构及参数**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
              ReLU-2           [-1, 32, 32, 32]               0
            Conv2d-3           [-1, 64, 32, 32]          18,496
              ReLU-4           [-1, 64, 32, 32]               0
         MaxPool2d-5           [-1, 64, 16, 16]               0
       BatchNorm2d-6           [-1, 64, 16, 16]             128
           Dropout-7           [-1, 64, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]          73,856
              ReLU-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
             ReLU-11          [-1, 128, 16, 16]               0
        MaxPool2d-12            [-1, 128, 8, 8]               0
      BatchNorm2d-13            [-1, 128, 8, 8]             256
          Dropout-14            [-1, 128, 8, 8]               0
           Conv2d-15            [-1, 256, 8, 8]         295,168
             ReLU-16            [-1, 256, 8, 8]               0
           Conv2d-17            [-1, 256, 8, 8]         590,080
             ReLU-18            [-1, 256, 8, 8]               0
        MaxPool2d-19            [-1, 256, 4, 4]               0
      BatchNorm2d-20            [-1, 256, 4, 4]             512
          Dropout-21            [-1, 256, 4, 4]               0
          Flatten-22                 [-1, 4096]               0
           Linear-23                 [-1, 1024]       4,195,328
             ReLU-24                 [-1, 1024]               0
          Dropout-25                 [-1, 1024]               0
           Linear-26                  [-1, 512]         524,800
             ReLU-27                  [-1, 512]               0
           Linear-28                   [-1, 10]           5,130
================================================================
Total params: 5,852,234
Trainable params: 5,852,234
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.72
Params size (MB): 22.32
Estimated Total Size (MB): 26.06
----------------------------------------------------------------
```

**（1）损失函数及优化器**

**损失函数定义为交叉熵函数**

**优化器使用Adam算法，学习率设为0.001，权重衰减设为0.0005**

```
criterion = nn.CrossEntropyLoss() # 交叉式损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0005) # 优化器
```

### 3、训练网络模型

**（1）进行15次epoch训练，每次epoch打印出训练集损失值及准确率，测试集损失值及准确率**

```
epoch : 1
training loss: 0.0049, acc 0.5513 
testing loss: 0.0038, testing acc 0.6567 
epoch : 2
training loss: 0.0031, acc 0.7188 
testing loss: 0.0028, testing acc 0.7570 
epoch : 3
training loss: 0.0025, acc 0.7774 
testing loss: 0.0025, testing acc 0.7879 
epoch : 4
training loss: 0.0022, acc 0.8096 
testing loss: 0.0024, testing acc 0.7830 
epoch : 5
training loss: 0.0019, acc 0.8305 
testing loss: 0.0023, testing acc 0.8050 
epoch : 6
training loss: 0.0017, acc 0.8461 
testing loss: 0.0022, testing acc 0.8170 
epoch : 7
training loss: 0.0016, acc 0.8569 
testing loss: 0.0020, testing acc 0.8321 
epoch : 8
training loss: 0.0015, acc 0.8712 
testing loss: 0.0019, testing acc 0.8411 
epoch : 9
training loss: 0.0014, acc 0.8802 
testing loss: 0.0019, testing acc 0.8413 
epoch : 10
training loss: 0.0013, acc 0.8871 
testing loss: 0.0017, testing acc 0.8532 
epoch : 11
training loss: 0.0012, acc 0.8929 
testing loss: 0.0019, testing acc 0.8466 
epoch : 12
training loss: 0.0011, acc 0.8981 
testing loss: 0.0018, testing acc 0.8506 
epoch : 13
training loss: 0.0011, acc 0.9011 
testing loss: 0.0018, testing acc 0.8491 
epoch : 14
training loss: 0.0010, acc 0.9096 
testing loss: 0.0018, testing acc 0.8563 
epoch : 15
training loss: 0.0010, acc 0.9105 
testing loss: 0.0020, testing acc 0.8485 
```

**（2）损失函数曲线**

![image-20211108194500357](D:\研\人工智能安全\homework\images\image-20211108194500357.png)

**（3）准确率曲线**

![image-20211108194611361](D:\研\人工智能安全\homework\images\image-20211108194611361.png)

### 4、测试

**（1）在10000张测试图片上分类的准确率为84.00 %**

```
Accuracy of the network on the 10000 test images: 84.00 %
```

**（2）在10000张测试图片上不同分类结果的准确率**

```
Accuracy for class airplane is: 90.4 %
Accuracy for class automobile is: 94.2 %
Accuracy for class bird  is: 72.2 %
Accuracy for class cat   is: 54.4 %
Accuracy for class deer  is: 80.6 %
Accuracy for class dog   is: 91.0 %
Accuracy for class frog  is: 90.6 %
Accuracy for class horse is: 91.0 %
Accuracy for class ship  is: 92.9 %
Accuracy for class truck is: 91.2 %
```

### 5、抽样部分结果并可视化显示

**可以观察到其中只有4个张图片分类错误，其它均分类正确**

```
Accuracy Rate = 85.15625%
```

![image-20211108195050243](D:\研\人工智能安全\homework\images\image-20211108195050243.png)

### 6、保存模型

**保存模型为当前文件夹下并命名为cifar_net.pth**

```
PATH = './modelcifar_net.pth'
torch.save(model.state_dict(), PATH)
```

