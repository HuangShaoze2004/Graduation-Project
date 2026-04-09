"""
数据加载与预处理模块
"""
import pandas as pd


def load_excel_data(file_path: str) -> pd.DataFrame:
    """
    读取 Excel 文件

    Args:
        file_path: Excel 文件路径

    Returns:
        DataFrame: 读取的数据
    """
    return pd.read_excel(file_path)


def fix_smiles(smiles: str) -> str:
    """
    修正 SMILES 字符串

    Args:
        smiles: 原始 SMILES 字符串

    Returns:
        str: 修正后的 SMILES 字符串
    """
    if smiles == 'ClC1=CC=NC2=C1C=CC3=C2N=CC=C3Cl':
        return 'ClC1=CC=NC2=C(C3=CC=C21)N=CC=C3Cl'
    return smiles


def build_ligand_cas_dict(df_lig: pd.DataFrame) -> dict:
    """
    建立配体编号到 CAS 号的映射字典

    Args:
        df_lig: 配体数据 DataFrame

    Returns:
        dict: 配体编号到 CAS 号的映射字典
    """
    return dict(zip(df_lig['No.'], df_lig['CAS']))


def preprocess_ligand_smiles(df_lig: pd.DataFrame) -> pd.DataFrame:
    """
    预处理配体 SMILES 数据

    Args:
        df_lig: 配体数据 DataFrame

    Returns:
        DataFrame: 预处理后的配体数据
    """
    df_lig = df_lig.copy()
    df_lig['SMILES'] = df_lig['SMILES'].astype(str).apply(fix_smiles)
    return df_lig
