# 贝叶斯优化催化剂 - 配体组合推荐

基于高斯过程回归的贝叶斯优化系统，用于预测和推荐最佳的催化剂 - 配体组合。

## 项目结构

```
bayes/
├── config/              # 配置文件
│   ├── __init__.py
│   └── settings.py      # 路径和参数配置
├── data/                # 数据处理
│   ├── __init__.py
│   ├── loader.py        # 数据加载
│   └── features.py      # 特征工程（Morgan 指纹）
├── models/              # 模型定义
│   ├── __init__.py
│   ├── gp_models.py     # 高斯过程回归模型
│   └── acquisition.py   # 采集函数（EI）
├── evaluation/          # 评估模块
│   ├── __init__.py
│   ├── metrics.py       # 评估指标
│   └── visualizer.py    # 可视化绘图
├── optimizer/           # 优化器
│   ├── __init__.py
│   └── bayesian_optimizer.py
├── iteration/           # 迭代训练管理
│   ├── __init__.py
│   └── manager.py
├── utils/               # 工具函数
│   ├── __init__.py
│   └── helpers.py
├── main.py              # 主入口
├── requirements.txt     # 依赖
└── README.md            # 使用说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 首次运行（冷启动模式）

```bash
python main.py
```

### 迭代训练模式

将新实验数据整理为 Excel 文件后，运行：

```bash
python main.py --iteration --new-data "新实验数据.xlsx"
```

或者简写：

```bash
python main.py -i -n "新实验数据.xlsx"
```

## 配置说明

编辑 `config/settings.py` 修改以下配置：

### 文件路径
- `PATH_CAT`: 催化剂数据文件路径
- `PATH_LIG`: 配体数据文件路径
- `PATH_COLD`: 冷启动数据文件路径
- `OUTPUT_FOLDER`: 输出文件夹路径

### 模型参数
- `MORGAN_RADIUS`: Morgan 指纹半径（默认 2）
- `MORGAN_FP_SIZE`: Morgan 指纹维度（默认 256）
- `NU`: Matern 核函数参数（默认 2.5）
- `NOISE_LEVEL`: 噪声水平（默认 0.1）
- `N_RESTARTS`: 核函数优化重启次数（默认 15）

### 筛选阈值
- `YIELD_THRESHOLD`: 产率阈值（默认 30.0%）
- `SELECTIVITY_THRESHOLD`: 选择性阈值（默认 4.0）
- `TOP_K`: 推荐组合数量（默认 50）

## 输出文件

运行后在输出文件夹生成以下文件：

1. **BO_Next_Batch.xlsx** - 推荐的下一轮实验组合
2. **Model_Train_Evaluation.png** - 模型训练评估图
3. **BO_Prediction_Space.png** - 预测空间分布图

## 新数据格式

用于迭代训练的新数据 Excel 文件应包含以下列：
- `Catalyst`: 催化剂名称
- `Name`: 配体编号
- `Yields%`: 产率
- `Selectivity(m:p)`: 选择性

## 工作流程

1. **冷启动阶段**: 读取初始实验数据训练模型
2. **预测阶段**: 对所有未测试的组合进行预测
3. **推荐阶段**: 基于 EI 采集函数筛选最佳组合
4. **实验验证**: 对推荐的组合进行实验
5. **迭代更新**: 将新数据加入训练集，重新训练模型

## 注意事项

- 确保输入的 Excel 文件格式正确
- SMILES 字符串需要符合 RDKit 规范
- 输出文件夹会自动创建
