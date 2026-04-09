"""
评估指标模块
"""
import numpy as np
from sklearn.metrics import r2_score


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 R²分数

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        float: R²分数
    """
    return r2_score(y_true, y_pred)
