## EfficientIR - 基于 EfficientNet 的图片检索工具

基于内容的图片检索方法有很多，即便是传统算法也能达到不错的效果。 本项目起源于要求使用传统算法（SIFT）完成的课程作业，但实现过程中遇到了一些影响实际使用的问题（例如建立索引慢）。 经过简单调研（指咨询群友和 Google）后发现 PC 上并没有什么好用的本地图片检索工具，于是干脆在作业外另外弄了这个小工具。  

使用到的主要工具是 EfficientNet 和 Hnswlib，使用前者在 ImageNet 上的预训练模型进行特征抽取，使用后者进行特征索引及检索。  

使用前请注意，如果有任何使用上的问题请尽量自行解决。  
比起花费时间维护，咱更希望像 digiKam/Eagle 这样的专业图像管理软件加入特性持续维护。  
如果您有任何新点子，欢迎 fork 实现后 pull request。

### 功能特性
 - 以图搜图
 - 图片相似度计算
 - 重复图片查找

### 依赖
 - Python 3.6+
 - numpy
 - onnxruntime
 - hnswlib
 - pillow
 - tqdm
 - onnx (若需更换模型则必要)
 - PyQt5

### 使用

#### GUI 使用

 > 当前 GUI 版本打包仅支持 Windows x86_64 环境使用

在 [Release](https://github.com/Sg4Dylan/EfficientIR/releases) 页面下载最新版程序打包，并解压到不包含中日韩文字的路径下。

初次使用需要建立索引：
1. 双击 `start.bat` 打开
2. 单击 `设置` - `添加索引目录` 添加需要索引的图库目录
3. 单击 `更新索引目录` 建立索引

搜索图片：
1. 单击 `...` 选择图片
2. 设置返回结果数量
3. 单击 `开始搜索`
4. 双击返回结果的文件路径打开图片

图片查重：
1. 单击 `查重` 切换到查重功能
2. 设置过滤结果使用到的相似度阈值（最低为 70%）
3. 单击 `开始查重`，并坐和放宽
4. 双击返回结果的文件路径打开图片

后续索引更新：
1. 单击 `设置` - `更新索引目录` 更新索引

### 更换模型

当前本 repo 已包含以下模型：
 - EfficientNet-B2
 - Once For All (flops@595M_top1@80.0_finetune@75)

关于将 PyTorch 或其他机器学习框架模型导出为 ONNX 模型的方法此处不再赘述。需要注意的是，导出的 ONNX 模型必须经过优化过程，可以使用 [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) （推荐）或本项目包含的 `opti.py`。

更换模型前可以使用 [Netron](https://lutzroeder.github.io/netron/) 检查模型的输入矩阵形状是否为 `1x3xNxN`,输出向量是否为 `1xN`。其中输入矩阵的形状对应 `efficient_ir.py` 中的常量 `img_size`，输出向量的形状对应 `init_index()` 方法的初始化维度 以及 `get_fv()` 方法的返回值。

若只是希望更换模型为更大的 EfficientNet 模型，那么只需要确认并修改 `efficient_ir.py` 中的 `img_size` 和 `model_path`。  

但如果需要更换为 Once For All 模型，虽然其输入与 EfficientNet-B2 相同，但输出是 `N` 并不是 `1xN`，故除修改 `model_path` 外，还需将 `get_fv()` 中返回所在行做出如下修改:  

```
修改前：
return self.session.run([], {self.model_input: norm_img_data})[0][0]
修改后
return self.session.run([], {self.model_input: norm_img_data})[0]
```

**注意：更换模型后一定要重新建立索引**

### GPU 加速

当前选择的模型均对性能进行了权衡，在支持 AVX 指令集的 CPU 上索引速度略高于甜品级 GPU。  
若期待通过 GPU 加速获得更好的性能及加速比，可以将模型换成更大规模的或增加索引时同时处理的图片数量。  
具体的操作请自行阅读并修改代码实现。  

如果是 NVIDIA 显卡，切换 GPU 推理的步骤：  
1. 安装 `onnxruntime-gpu` ；  
2. 取消 `efficient_ir.py` 第 61 行的注释；  
3. 将 provider 需要换成 `GPUExecutionProvider`。

支持 DX12 Compute 的任意显卡（包括集成显卡），切换 GPU 推理的步骤：  
1. 安装 `onnxruntime-dml` ；
2. 取消 `efficient_ir.py` 第 61 行的注释。  

### Q&A

> Q：可承载最大索引数量是多少？如何修改？  
> A：目前是 1000000。可以在 `efficient_ir.py` 中修改，数值将在下一次加载时生效。

> Q：检索效果不佳怎么解决？  
> A：当前代码中使用 EfficientNet-b2 模型是经过权衡后决定的，若追求更佳检索效果请自行更换更大规模的 EfficientNet 模型或其他的 SOTA 模型。本项目将持续关注 SOTA 模型的发展，并在 [Wiki](https://github.com/Sg4Dylan/EfficientIR/wiki) 中更新相关测试结果。

### TODO

 - [x] 实现 GUI 工具
 - [ ] 移植到 C++ 使用
 - [ ] 替换 digiKam 内的模糊搜索

### References
 - [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
 - [Once For All](https://github.com/mit-han-lab/once-for-all)
 - [Hnswlib](https://github.com/nmslib/hnswlib)
