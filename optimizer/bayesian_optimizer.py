"""
贝叶斯优化器模块
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from models.gp_models import create_gp_model, train_model, predict
from models.acquisition import calculate_ei
from config.settings import YIELD_THRESHOLD, SELECTIVITY_THRESHOLD, TOP_K


class BayesianOptimizer:
    """贝叶斯优化器类"""

    def __init__(self):
        self.model_yield = None
        self.model_sel = None
        self.cat_dict = {}
        self.lig_dict = {}
        self.lig_cas_dict = {}

    def set_feature_dicts(self, cat_dict: Dict, lig_dict: Dict, lig_cas_dict: Dict):
        """
        设置特征字典

        Args:
            cat_dict: 催化剂特征字典
            lig_dict: 配体特征字典
            lig_cas_dict: 配体编号到 CAS 号的映射字典
        """
        self.cat_dict = cat_dict
        self.lig_dict = lig_dict
        self.lig_cas_dict = lig_cas_dict

    def build_training_set(
        self,
        df_cold: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建训练集

        Args:
            df_cold: 冷启动数据 DataFrame

        Returns:
            X_train, y_yield, y_sel
        """
        X_train, y_yield, y_sel = [], [], []

        for _, row in df_cold.iterrows():
            c_name, l_name = row['Catalyst'], row['Name']
            if c_name in self.cat_dict and l_name in self.lig_dict:
                X_train.append(np.concatenate([self.cat_dict[c_name], self.lig_dict[l_name]]))
                y_yield.append(row['Yields%'])
                y_sel.append(row['Selectivity(m:p)'])

        return np.array(X_train), np.array(y_yield), np.array(y_sel)

    def train_models(self, X_train: np.ndarray, y_yield: np.ndarray, y_sel: np.ndarray):
        """
        训练模型

        Args:
            X_train: 特征矩阵
            y_yield: 产率目标值
            y_sel: 选择性目标值
        """
        self.model_yield = create_gp_model()
        self.model_sel = create_gp_model()

        self.model_yield = train_model(self.model_yield, X_train, y_yield)
        self.model_sel = train_model(self.model_sel, X_train, y_sel)

    def predict_pool(
        self,
        X_pool: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        预测候选池

        Args:
            X_pool: 候选池特征矩阵

        Returns:
            mu_y, std_y, mu_s, std_s
        """
        mu_y, std_y = predict(self.model_yield, X_pool, return_std=True)
        mu_s, std_s = predict(self.model_sel, X_pool, return_std=True)
        return mu_y, std_y, mu_s, std_s

    def generate_untested_combos(
        self,
        df_cold: pd.DataFrame
    ) -> Tuple[list, np.ndarray]:
        """
        生成未测试的组合池

        Args:
            df_cold: 冷启动数据 DataFrame

        Returns:
            all_combos, X_pool
        """
        all_combos = []
        X_pool = []
        tested_set = set(zip(df_cold['Catalyst'], df_cold['Name']))

        for c_name, c_fp in self.cat_dict.items():
            for l_name, l_fp in self.lig_dict.items():
                if (c_name, l_name) not in tested_set:
                    all_combos.append((c_name, l_name))
                    X_pool.append(np.concatenate([c_fp, l_fp]))

        return all_combos, np.array(X_pool)

    def screen_recommendations(
        self,
        all_combos: list,
        mu_y: np.ndarray,
        mu_s: np.ndarray,
        std_y: np.ndarray,
        std_s: np.ndarray,
        y_yield: np.ndarray,
        y_sel: np.ndarray,
        top_k: int = None
    ) -> pd.DataFrame:
        """
        筛选推荐结果

        Args:
            all_combos: 所有组合列表
            mu_y: 产率预测均值
            mu_s: 选择性预测均值
            std_y: 产率预测标准差
            std_s: 选择性预测标准差
            y_yield: 训练集产率
            y_sel: 训练集选择性
            top_k: 推荐数量

        Returns:
            pd.DataFrame: 推荐结果
        """
        if top_k is None:
            top_k = TOP_K

        # 计算 EI
        ei_y = calculate_ei(mu_y, std_y, np.max(y_yield))
        ei_s = calculate_ei(mu_s, std_s, np.max(y_sel))

        # 构建结果 DataFrame
        results = pd.DataFrame({
            'Catalyst': [item[0] for item in all_combos],
            'Ligand': [item[1] for item in all_combos],
            'Ligand_CAS': [self.lig_cas_dict.get(item[1], None) for item in all_combos],
            'Pred_Yield(%)': mu_y,
            'Pred_Sel(m:p)': mu_s,
            'EI_Yield': ei_y,
            'EI_Sel': ei_s
        })

        results['Total_Score'] = results['EI_Yield'] * results['EI_Sel']

        # 筛选高潜力组合 (p:m > SELECTIVITY_THRESHOLD, 即 m:p < 1/SELECTIVITY_THRESHOLD)
        top_hits = results[
            (results['Pred_Yield(%)'] > YIELD_THRESHOLD) &
            (results['Pred_Sel(m:p)'] < (1 / SELECTIVITY_THRESHOLD))
        ]

        # 获取推荐列表
        if len(top_hits) < top_k:
            recommendations = results.sort_values(by='Total_Score', ascending=False).head(top_k)
        else:
            recommendations = top_hits.sort_values(by='Total_Score', ascending=False).head(top_k)

        return recommendations, results
