# PySTModels
用于阅读论文并学习代码使用，主要为时空数据学习。

## 文件结构
```
—  PySTModels
|- data      : 存放数据集
|- model     : 模型目录
|- utils     : 功能函数
  |- dataset_conf       : 数据集读取逻辑
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