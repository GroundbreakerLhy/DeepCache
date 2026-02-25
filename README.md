# DeepCache

Remotely recover (steal) DNN model architectures from DNN executables with
cache side channel.

## Repository Structure
```
|- embedding/
  |- models/
    |- __init__.py
    |- normalize.py           # 模型归一化工具。
    |- resnet_cache.py        # 为缓存处理适配的 ResNet 模型。
  |- BatchAverage.py        # 定义 BatchCriterion 类，实现基于负采样的批处理损失计算。
  |- CacheDataset.py        # 定义数据集类（例如 LargeCachePicDataset），从跟踪文件中生成缓存图片用于训练嵌入模型。
  |- Embed.py               # 嵌入模型的核心文件，使用无监督学习从跟踪数据中提取嵌入向量。包括数据集加载、模型构建、训练和数据库匹配功能。
  |- embedding_test.py      # 训练和评估嵌入模型的测试脚本，支持 CIFAR 数据集和自定义参数。
  |- utils.py               # 嵌入相关的工具函数，如kNN标签生成和聚类评估。
|- encoder/
  |- model/
    |- conv_encoder.py        # 定义 CNN 编码器和 RED 模型，包括注意力机制和 ConvLSTM 集成。
    |- convolution_lstm.py    # 实现 ConvLSTM 单元用于序列建模。
  |- utils/
    |- __init__.py
    |- data.py                # 数据加载函数，从 .npy 文件中加载训练/测试数据。
    |- matrix_generator.py    # 生成签名矩阵的工具，从跟踪数据中创建节点签名。
  |- encoder_test.py        # 训练和评估编码器模型的测试脚本。
  |- Encoder.py             # 编码器模块，定义 RED（循环编码器-解码器）网络用于处理跟踪数据、生成签名矩阵和训练模型。
|- image/
  |- tmp.jpg
  |- visualize.py           # 将缓存跟踪数据可视化为图像（PIL Image）用于调试和分析。
|- pin_tool/
  |- InstLogger.cpp         # 指令记录器。
  |- ObfusSim.cpp           # 混淆模拟器。
  |- ORAMTrace.cpp          # ORAM（Oblivious RAM）跟踪记录器。
  |- TraceLogger.cpp        # PIN工具源码，用于记录内存访问跟踪（缓存状态），支持函数范围过滤。
  |- TraceLogger_kernel.cpp # 类似于 TraceLogger，但用于内核模式。
|- preprocess/
  |- onnx_info.py           # 处理ONNX模型，提取输入/输出信息用于模型分析。
  |- split_funcs.py         # 从汇编代码中分割和清理函数，用于提取函数边界。
|- attr_labels_glow.json  # 数据文件，存储 Glow 和 TVM 编译器的属性标签，用于模型匹配。
|- attr_labels_tvm.json   # 数据文件，存储 Glow 和 TVM 编译器的属性标签，用于模型匹配。
|- config.py              # 配置文件，定义 PIN 工具的路径（例如 pin_home 和 mypintool_dir）用于动态二进制插桩。
|- dataset.py             # 定义 CacheDataset 类，用于加载和处理缓存跟踪数据（从.log文件中读取向量序列）。
|- main.py                # 主入口脚本，定义两个实验函数 experiment_tvm_O3() 和 experiment_glow()，分别用于处理 TVM 和 Glow 编译器的 DNN 模型恢复。包括嵌入模型训练、数据库匹配和循环因子检查。
|- pin_logger.py          # 运行PIN工具的封装脚本，定义记录跟踪的命令行（例如 TraceLogger.so 和 TraceLogger_kernel.so），并提供执行函数。
|- README.md              # 此README文件。
|- utils.py               # 工具函数库，包括JSON读写、函数地址范围检索、循环因子检查等。导入 pin_logger 和 encoder.Encoder。
```