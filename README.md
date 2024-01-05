# Hyper-SegUNet:基于超网络的超参自学习医学图像分割模型
郭逸凡1 裴瑄1 王大寒1,2 陈培芝1,2* 

（1.厦门理工学院 计算机与信息工程学院 福建 厦门 361024； 2.福建省模式识别与图像理解重点实验室 福建 厦门 361024）

## Citations
[1]郭逸凡,裴瑄,王大寒等.Hyper-SegUNet:基于超网络的超参自学习医学图像分割模型[J].四川师范大学学报(自然科学版),2024,47(01):127-135.
```bibtex
@article{郭2024hypersegunet,
  title={Hyper-SegUNet:基于超网络的超参自学习医学图像分割模型},
  author={郭逸凡 and 裴瑄 and 王大寒 and 陈培芝},
  journal={四川师范大学学报(自然科学版)},
  volume = {47(01)},
  pages = {127-135},
  year={2024}
}
```
# 安装环境和运行

安装conda虚拟环境
```
conda create -n hypersegunet python=3.6 
```
激活环境
```
conda activate hypersegunet
```
从torch官网选择对应版本的torch， 这根据你的服务器具体配置有所不同
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
安装依赖库
```
pip install tqdm h5py opencv-python scipy SimpleITK==2.0.0 medpy monai tensorboardX
```

训练
```
train_data：训练数据所在的路径

--dataset：数据集选择，默认是Synapse

--max_epoch：最大训练轮数

--batch_size：批量大小

--base_lr：学习率

--img_size：图像尺寸

--hyper_layer_num 超网络中全连接层的数量

--hyper_layer_size：超网络中全连接层的维度

--save_pth_dir 保存的模型文件的路径

--mixup 是否使用mixup策略（未来进一步扩展的工作）

--homo 是否使用同方差不确定性策略（未来进一步扩展的工作）
```

训练
```
python train.py --train_data /home/gyf/MyData/Synapse/Synapse/train_npz/ --save_pth_dir ./save_model_pt/Synapse/Hyper_model/
```

测试 注意这里修改测试集的路径
```
python test.py
```



# 致谢
非常感谢TransUNet的作者Jieneng Chen等人，向我们提供处理好的数据集。

非常感谢HyperMorph的作者Andrew Hoopes等人、Hypernet的作者Sungmin Hong等人的开源项目，受他们的启发，我们才能做出本文工作。


# 其他
你可以在BiliBili和知乎联系我，任何有关医学图像配准和医学图像处理的问题，力所能及，都可以咨询。

BiliBili（暑假作业多了）：https://m.bilibili.com/space/1655726

知乎（妹儿红着呢）：https://www.zhihu.com/people/GhostOgre
