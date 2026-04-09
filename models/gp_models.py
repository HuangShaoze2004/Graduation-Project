"""
高斯过程回归模型模块
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from config.settings import NU, NOISE_LEVEL, N_RESTARTS, RANDOM_STATE, CV_SPLITS, CV_SHUFFLE, CV_RANDOM_STATE


def create_gp_model(random_state: int = None) -> GaussianProcessRegressor:
    """
    创建高斯过程回归模型

    Args:
        random_state: 随机种子

    Returns:
        GaussianProcessRegressor: 高斯过程回归模型
    """
    if random_state is None:
        random_state = RANDOM_STATE

    kernel = Matern(nu=NU) + WhiteKernel(noise_level=NOISE_LEVEL)

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS,
        normalize_y=True,
        random_state=random_state
    )
    return model


def train_model(model: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    训练模型

    Args:
        model: 高斯过程回归模型
        X: 特征矩阵
        y: 目标值

    Returns:
        GaussianProcessRegressor: 训练好的模型
    """
    model.fit(X, y)
    return model


def predict(model: GaussianProcessRegressor, X: np.ndarray, return_std: bool = False):
    """
    使用模型进行预测

    Args:
        model: 训练好的模型
        X: 特征矩阵
        return_std: 是否返回标准差

    Returns:
        预测值（和标准差）
    """
    return model.predict(X, return_std=return_std)


def cross_validate_models(
    X: np.ndarray,
    y_yield: np.ndarray,
    y_sel: np.ndarray,
    random_state: int = None
) -> dict:
    """
    交叉验证评估模型

    Args:
        X: 特征矩阵
        y_yield: 产率目标值
        y_sel: 选择性目标值
        random_state: 随机种子

    Returns:
        dict: 包含两个模型的交叉验证结果
    """
    if random_state is None:
        random_state = CV_RANDOM_STATE

    kf = KFold(
        n_splits=CV_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=random_state
    )

    model_yield = create_gp_model(random_state)
    model_sel = create_gp_model(random_state)

    cv_yield = cross_validate(model_yield, X, y_yield, cv=kf, scoring='r2')
    cv_sel = cross_validate(model_sel, X, y_sel, cv=kf, scoring='r2')

    return {
        'yield': {
            'mean': cv_yield['test_score'].mean(),
            'std': cv_yield['test_score'].std()
        },
        'selectivity': {
            'mean': cv_sel['test_score'].mean(),
            'std': cv_sel['test_score'].std()
        }
    }
