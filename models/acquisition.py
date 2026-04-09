"""
采集函数模块 - 期望提升 (Expected Improvement) 计算
"""
import numpy as np
from scipy.stats import norm

from config.settings import XI


def calculate_ei(mu: np.ndarray, std: np.ndarray, y_best: float) -> np.ndarray:
    """
    计算期望提升 (Expected Improvement)

    Args:
        mu: 预测均值
        std: 预测标准差
        y_best: 当前最佳值

    Returns:
        np.ndarray: 每个样本的 EI 值
    """
    with np.errstate(divide='ignore'):
        imp = mu - y_best - XI
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
    return ei
