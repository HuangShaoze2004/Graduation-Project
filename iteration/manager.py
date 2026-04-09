"""
迭代训练管理模块
管理下一轮训练数据的更新和合并
"""
import os
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime


class IterationManager:
    """迭代训练管理器"""

    def __init__(self, cold_start_path: str, output_folder: str):
        """
        初始化迭代管理器

        Args:
            cold_start_path: 冷启动数据文件路径
            output_folder: 输出文件夹路径
        """
        self.cold_start_path = cold_start_path
        self.output_folder = output_folder
        self.recommendations_path = os.path.join(output_folder, "BO_Next_Batch.xlsx")

    def load_previous_recommendations(self) -> Optional[pd.DataFrame]:
        """
        加载上一轮的推荐结果

        Returns:
            DataFrame or None: 推荐结果 DataFrame，如果文件不存在则返回 None
        """
        if os.path.exists(self.recommendations_path):
            return pd.read_excel(self.recommendations_path)
        return None

    def check_new_data_file(self, new_data_path: str) -> bool:
        """
        检查新数据文件是否存在

        Args:
            new_data_path: 新数据文件路径

        Returns:
            bool: 文件是否存在
        """
        return os.path.exists(new_data_path)

    def merge_new_data(
        self,
        df_cold: pd.DataFrame,
        df_new: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将新数据合并到冷启动数据中

        Args:
            df_cold: 冷启动数据
            df_new: 新实验数据

        Returns:
            DataFrame: 合并后的数据
        """
        # 确保列名一致
        required_cols = ['Catalyst', 'Name', 'Yields%', 'Selectivity(m:p)']
        for col in required_cols:
            if col not in df_new.columns:
                raise ValueError(f"新数据缺少必需列：{col}")

        # 合并数据（去除重复）
        df_combined = pd.concat([df_cold, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Catalyst', 'Name'], keep='last')

        return df_combined

    def load_cold_start_data(self) -> pd.DataFrame:
        """
        加载冷启动数据

        Returns:
            DataFrame: 冷启动数据
        """
        if os.path.exists(self.cold_start_path):
            return pd.read_excel(self.cold_start_path)
        raise FileNotFoundError(f"冷启动数据文件不存在：{self.cold_start_path}")

    def save_updated_data(
        self,
        df_updated: pd.DataFrame,
        suffix: str = None
    ) -> str:
        """
        保存更新后的数据

        Args:
            df_updated: 更新后的数据
            suffix: 文件名后缀（可选）

        Returns:
            str: 保存路径
        """
        if suffix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_updated_{timestamp}"

        filename = f"冷启动数据{suffix}.xlsx"
        save_path = os.path.join(self.output_folder, filename)

        df_updated.to_excel(save_path, index=False)
        print(f"更新后的数据已保存至：{save_path}")

        return save_path

    def get_already_recommended_set(self) -> set:
        """
        获取已推荐的组合集合

        Returns:
            set: (Catalyst, Ligand) 元组集合
        """
        recommendations = self.load_previous_recommendations()
        if recommendations is not None:
            return set(zip(recommendations['Catalyst'], recommendations['Ligand']))
        return set()

    def create_iteration_report(
        self,
        df_cold: pd.DataFrame,
        df_updated: pd.DataFrame,
        n_recommendations: int
    ) -> str:
        """
        创建迭代报告

        Args:
            df_cold: 原始冷启动数据
            df_updated: 更新后的数据
            n_recommendations: 推荐组合数量

        Returns:
            str: 报告内容
        """
        n_before = len(df_cold)
        n_after = len(df_updated)

        report = f"""
==================== 迭代训练报告 ====================
训练数据更新:
  - 更新前数据量：{n_before} 条
  - 更新后数据量：{n_after} 条
  - 新增数据量：{n_after - n_before} 条

推荐结果:
  - 推荐组合数量：{n_recommendations} 条

说明:
  - 请将上一轮推荐的组合进行实验
  - 将实验结果（产率和选择性）添加到冷启动数据中
  - 重新运行本脚本进行下一轮优化
====================================================
"""
        return report
