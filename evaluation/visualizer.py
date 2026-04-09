"""
可视化模块
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from config.settings import (
    OUTPUT_TRAIN_EVAL_IMAGE,
    OUTPUT_PREDICTION_SPACE_IMAGE,
    YIELD_THRESHOLD,
    SELECTIVITY_THRESHOLD
)


def plot_training_evaluation(
    y_yield: np.ndarray,
    y_pred_train_yield: np.ndarray,
    y_sel: np.ndarray,
    y_pred_train_sel: np.ndarray,
    r2_train_yield: float,
    r2_train_sel: float,
    output_folder: str
) -> str:
    """
    绘制训练集评估图

    Args:
        y_yield: 实际产率值
        y_pred_train_yield: 预测产率值
        y_sel: 实际选择性值
        y_pred_train_sel: 预测选择性值
        r2_train_yield: 产率 R²分数
        r2_train_sel: 选择性 R²分数
        output_folder: 输出文件夹路径

    Returns:
        str: 保存图片路径
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(
        y_yield, y_pred_train_yield,
        alpha=0.7, color='royalblue', edgecolor='k', s=50,
        label=f'R² = {r2_train_yield:.3f}'
    )
    axes[0].plot(
        [min(y_yield), max(y_yield)],
        [min(y_yield), max(y_yield)],
        'r--', linewidth=2, label='Ideal Fit (y=x)'
    )
    axes[0].set_xlabel('Measured Yields (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Yields (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Yields: Model Performance', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=11, frameon=True)

    axes[1].scatter(
        y_sel, y_pred_train_sel,
        alpha=0.7, color='forestgreen', edgecolor='k', s=50,
        label=f'R² = {r2_train_sel:.3f}'
    )
    axes[1].plot(
        [min(y_sel), max(y_sel)],
        [min(y_sel), max(y_sel)],
        'r--', linewidth=2, label='Ideal Fit (y=x)'
    )
    axes[1].set_xlabel('Measured Selectivity (m:p)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Selectivity (m:p)', fontsize=12, fontweight='bold')
    axes[1].set_title('Selectivity: Model Performance', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=11, frameon=True)

    plt.tight_layout()
    eval_img_path = os.path.join(output_folder, OUTPUT_TRAIN_EVAL_IMAGE)
    plt.savefig(eval_img_path, dpi=300)
    plt.close()

    return eval_img_path


def plot_prediction_space(
    results: 'pd.DataFrame',
    recommendations: 'pd.DataFrame',
    top_k: int,
    output_folder: str
) -> str:
    """
    绘制预测空间分布图

    Args:
        results: 所有预测结果 DataFrame
        recommendations: 推荐组合 DataFrame
        top_k: 推荐数量
        output_folder: 输出文件夹路径

    Returns:
        str: 保存图片路径
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        results['Pred_Yield(%)'], results['Pred_Sel(m:p)'],
        c='lightgray', alpha=0.6, s=30, label='All Predicted Combos'
    )

    ax.scatter(
        recommendations['Pred_Yield(%)'], recommendations['Pred_Sel(m:p)'],
        c='crimson', marker='*', s=150, edgecolor='black', zorder=5,
        label=f'Top {top_k} Recommended'
    )

    ax.axvline(
        x=YIELD_THRESHOLD, color='royalblue', linestyle='--', linewidth=2, zorder=1,
        label=f'Target Yield ({YIELD_THRESHOLD}%)'
    )
    ax.axhline(
        y=SELECTIVITY_THRESHOLD, color='forestgreen', linestyle='--', linewidth=2, zorder=1,
        label=f'Target Selectivity ({SELECTIVITY_THRESHOLD})'
    )

    ax.set_title(
        'Bayesian Optimization: Dual-Objective Prediction Space',
        fontsize=16, fontweight='bold', pad=15
    )
    ax.set_xlabel('Predicted Yield (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Selectivity (m:p)', fontsize=14, fontweight='bold')

    x_min, x_max = min(results['Pred_Yield(%)'].min(), 0), max(results['Pred_Yield(%)'].max(), 100)
    y_min, y_max = min(results['Pred_Sel(m:p)'].min(), 0), max(results['Pred_Sel(m:p)'].max(), 6.0)
    ax.set_xlim(x_min - 5, x_max + 10)
    ax.set_ylim(y_min - 0.5, y_max + 1.0)

    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.7)

    img_path = os.path.join(output_folder, OUTPUT_PREDICTION_SPACE_IMAGE)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    return img_path
