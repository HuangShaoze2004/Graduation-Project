"""
贝叶斯优化配置文件
包含路径设置、模型参数、筛选阈值等配置
"""
import os

# ================= 文件路径配置 =================
# 输入数据路径
PATH_CAT = r"C:\Users\HuangShaoze\Desktop\BO\化学空间\catalysts.xlsx"
PATH_LIG = r"C:\Users\HuangShaoze\Desktop\BO\化学空间\N_ligands.xlsx"
PATH_COLD = r"C:\Users\HuangShaoze\Desktop\BO\迭代数据\冷启动数据.xlsx"

# 输出文件夹路径
OUTPUT_FOLDER = r"C:\Users\HuangShaoze\Desktop\BO\下一轮预测"

# ================= 模型参数配置 =================
# Morgan 指纹参数
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 256

# 高斯过程核函数参数
NU = 2.5  # Matern 核函数的 nu 参数
NOISE_LEVEL = 0.1  # WhiteKernel 噪声水平
N_RESTARTS = 15  # 核函数优化重启次数
RANDOM_STATE = 42  # 随机种子

# 交叉验证参数
CV_SPLITS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42

# ================= 采集函数参数 =================
XI = 0.01  # EI 计算中的探索参数

# ================= 筛选阈值 =================
YIELD_THRESHOLD = 30.0  # 产率阈值 (%)
SELECTIVITY_THRESHOLD = 2.0  # 选择性阈值 (p:m)
TOP_K = 24  # 推荐组合数量

# ================= 输出文件名 =================
OUTPUT_TRAIN_EVAL_IMAGE = "Model_Train_Evaluation.png"
OUTPUT_PREDICTION_SPACE_IMAGE = "BO_Prediction_Space.png"
OUTPUT_RECOMMENDATIONS = "BO_Next_Batch.xlsx"
OUTPUT_COLD_START_UPDATED = "冷启动数据_更新后.xlsx"


def ensure_output_folder():
    """确保输出文件夹存在"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    return OUTPUT_FOLDER
