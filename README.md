# PySTModels
用于阅读论文并学习代码使用，主要为时空数据学习。

## 文件结构
```
—  PySTModels
|- data      : 存放数据集
|- model     : 模型目录
|- utils     : 功能函数
  |- dataProvider       : 数据集读取逻辑(BasicDataProvider作为基类实现任意数据集读取策略)
    > api: transform()、inverse_transform()
  |- base_function.py   : 基本功能函数
  |- data_function.py   : 数据相关接口函数
|- Run.py    : 接口函数（用于调用其他函数）
```


## 模型与论文列表
- iTransformer: 
  - Paper: https://arxiv.org/abs/2310.06625 (ICLR 2024 A)
  - Code:  https://github.com/thuml/iTransformer




# 框架细节：
## utils:
### dataProvider

- init():



# 测试list
- 多GPU
- linux
- wsl windows


# TODO LIST:
1. 数据在内存中被复制为了5份（all、train、valid、test、train-valid），约数据集大小的三倍使用量，后续工作应平衡读取速度和内存使用。