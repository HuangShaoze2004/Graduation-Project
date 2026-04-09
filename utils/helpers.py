"""
工具函数模块
"""
import os
import pandas as pd
from typing import List


def ensure_output_folder(output_folder: str) -> str:
    """
    确保输出文件夹存在

    Args:
        output_folder: 输出文件夹路径

    Returns:
        str: 输出文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def save_results(
    recommendations: pd.DataFrame,
    output_path: str,
    output_cols: List[str] = None
) -> str:
    """
    保存结果到 Excel 文件

    Args:
        recommendations: 推荐结果 DataFrame
        output_path: 输出路径
        output_cols: 输出列名列表

    Returns:
        str: 保存路径
    """
    if output_cols is None:
        output_cols = [
            'Catalyst', 'Ligand', 'Ligand_CAS',
            'Pred_Yield(%)', 'Pred_Sel(m:p)', 'Total_Score'
        ]

    recommendations_output = recommendations[output_cols]
    recommendations_output.to_excel(output_path, index=False)
    return output_path
