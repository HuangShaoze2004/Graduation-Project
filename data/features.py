"""
特征工程模块 - Morgan 指纹生成
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

from config.settings import MORGAN_RADIUS, MORGAN_FP_SIZE


def get_morgan_fingerprint(name: str, smiles: str, mol_type: str) -> np.ndarray:
    """
    提取分子指纹

    Args:
        name: 分子名称
        smiles: SMILES 字符串
        mol_type: 分子类型（催化剂/配体）

    Returns:
        np.ndarray: Morgan 指纹数组
    """
    # 创建指纹生成器
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE
    )

    # 临时禁用 RDKit 日志
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    mol = Chem.MolFromSmiles(smiles)

    lg.setLevel(RDLogger.WARNING)

    if mol:
        return np.array(fp_gen.GetFingerprintAsNumPy(mol))
    else:
        print(f"⚠️ 警告：无法解析 {mol_type} [{name}] 的 SMILES: {smiles}")
        print(f"   -> 请在 Excel 表格中核对并修复该物质的结构！\n")
        return np.zeros(MORGAN_FP_SIZE)


def build_feature_dict(df: pd.DataFrame, name_col: str, smiles_col: str, mol_type: str) -> dict:
    """
    构建特征字典

    Args:
        df: 数据 DataFrame
        name_col: 名称列名
        smiles_col: SMILES 列名
        mol_type: 分子类型

    Returns:
        dict: 名称到指纹的映射字典
    """
    feature_dict = {}
    for _, row in df.iterrows():
        feature_dict[row[name_col]] = get_morgan_fingerprint(
            row[name_col],
            str(row[smiles_col]),
            mol_type
        )
    return feature_dict
