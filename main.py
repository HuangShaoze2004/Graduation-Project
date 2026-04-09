"""
贝叶斯优化主入口
用于催化剂 - 配体组合推荐和迭代训练
"""
import os
import sys
import argparse
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    PATH_CAT, PATH_LIG, PATH_COLD, OUTPUT_FOLDER,
    TOP_K, OUTPUT_RECOMMENDATIONS
)
from config import ensure_output_folder
from data.loader import load_excel_data, preprocess_ligand_smiles, build_ligand_cas_dict
from data.features import build_feature_dict
from models.gp_models import cross_validate_models
from evaluation.metrics import calculate_r2
from evaluation.visualizer import plot_training_evaluation, plot_prediction_space
from optimizer import BayesianOptimizer
from iteration import IterationManager
from utils.helpers import save_results


def main(iteration_mode: bool = False, new_data_path: str = None):
    """
    主函数

    Args:
        iteration_mode: 是否启用迭代模式
        new_data_path: 新数据文件路径（迭代模式用）
    """
    print("=" * 60)
    print("贝叶斯优化 - 催化剂 - 配体组合推荐")
    print("=" * 60)

    # 确保输出文件夹存在
    output_folder = ensure_output_folder()
    print(f"\n输出文件夹：{output_folder}")

    # ================= 迭代模式处理 =================
    iteration_manager = IterationManager(PATH_COLD, output_folder)

    if iteration_mode and new_data_path:
        print("\n[迭代模式] 检测新数据文件...")
        if iteration_manager.check_new_data_file(new_data_path):
            print("正在加载新数据...")
            df_new = pd.read_excel(new_data_path)
            df_cold = iteration_manager.load_cold_start_data()
            df_cold = iteration_manager.merge_new_data(df_cold, df_new)
            print(f"数据已合并：{len(df_cold)} 条记录")
        else:
            print(f"警告：新数据文件不存在 ({new_data_path})，使用原始冷启动数据")
            df_cold = iteration_manager.load_cold_start_data()
    else:
        print("\n正在加载冷启动数据...")
        df_cold = iteration_manager.load_cold_start_data()

    # ================= 数据加载 =================
    print("\n正在加载催化剂和配体数据...")
    df_cat = load_excel_data(PATH_CAT)
    df_lig = load_excel_data(PATH_LIG)

    # 预处理配体 SMILES
    df_lig = preprocess_ligand_smiles(df_lig)

    # 建立配体编号到 CAS 号的映射
    lig_cas_dict = build_ligand_cas_dict(df_lig)

    # ================= 特征工程 =================
    print("\n正在生成分子指纹特征 (256 位 Morgan 指纹)...")
    cat_dict = build_feature_dict(df_cat, 'Name', 'SMILES', "催化剂")
    lig_dict = build_feature_dict(df_lig, 'No.', 'SMILES', "配体")

    # ================= 初始化优化器 =================
    optimizer = BayesianOptimizer()
    optimizer.set_feature_dicts(cat_dict, lig_dict, lig_cas_dict)

    # ================= 构建训练集 =================
    print("\n正在构建训练集...")
    X_train, y_yield, y_sel = optimizer.build_training_set(df_cold)
    print(f"训练集大小：{len(X_train)} 个组合")

    # ================= 模型评估 =================
    print("\n[后台运行] 正在进行交叉验证客观评估...")
    cv_results = cross_validate_models(X_train, y_yield, y_sel)
    print(
        f"后台参考 -> [产率] CV R²: {cv_results['yield']['mean']:.3f} ± {cv_results['yield']['std']:.3f} | "
        f"[选择性] CV R²: {cv_results['selectivity']['mean']:.3f} ± {cv_results['selectivity']['std']:.3f}"
    )

    # ================= 训练模型 =================
    print("\n正在使用全量冷启动数据训练模型...")
    optimizer.train_models(X_train, y_yield, y_sel)

    # 训练集预测和评估
    y_pred_train_yield = optimizer.model_yield.predict(X_train)
    y_pred_train_sel = optimizer.model_sel.predict(X_train)

    r2_train_yield = calculate_r2(y_yield, y_pred_train_yield)
    r2_train_sel = calculate_r2(y_sel, y_pred_train_sel)

    print(f"[产率 Yields%] 训练集 R²: {r2_train_yield:.3f}")
    print(f"[选择性 Sel(m:p)] 训练集 R²: {r2_train_sel:.3f}")

    # ================= 绘制训练评估图 =================
    print("\n正在绘制训练集评估图...")
    eval_img_path = plot_training_evaluation(
        y_yield, y_pred_train_yield,
        y_sel, y_pred_train_sel,
        r2_train_yield, r2_train_sel,
        output_folder
    )
    print(f"模型评估拟合效果图已保存至：{eval_img_path}")

    # ================= 生成未测试组合池 =================
    print("\n正在遍历化学空间进行预测...")
    all_combos, X_pool = optimizer.generate_untested_combos(df_cold)
    print(f"未测试组合数量：{len(all_combos)}")

    # ================= 预测和采集函数计算 =================
    mu_y, std_y, mu_s, std_s = optimizer.predict_pool(X_pool)

    # ================= 筛选推荐结果 =================
    print("\n正在计算期望提升 (EI) 并筛选推荐...")
    recommendations, results = optimizer.screen_recommendations(
        all_combos, mu_y, mu_s, std_y, std_s, y_yield, y_sel, TOP_K
    )

    # ================= 保存推荐结果 =================
    output_path = os.path.join(output_folder, OUTPUT_RECOMMENDATIONS)
    save_results(recommendations, output_path)
    print(f"\n推荐列表已生成！排名前 {TOP_K} 的高潜力组合已保存至：{output_path}")

    print("\n为您推荐的下一轮实验组合 (Top 5 预览)：")
    preview_cols = ['Catalyst', 'Ligand', 'Ligand_CAS', 'Pred_Yield(%)', 'Pred_Sel(m:p)']
    print(recommendations[preview_cols].head())

    # ================= 绘制预测空间分布图 =================
    print("\n正在生成预测空间分布可视化图片...")
    space_img_path = plot_prediction_space(
        results, recommendations, TOP_K, output_folder
    )
    print(f"化学空间分布图片已保存至：{space_img_path}")

    # ================= 迭代报告 =================
    if iteration_mode:
        print("\n" + iteration_manager.create_iteration_report(
            df_cold, df_cold, len(recommendations)
        ))

    print("\n" + "=" * 60)
    print("贝叶斯优化完成！")
    print("=" * 60)

    return recommendations


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='贝叶斯优化 - 催化剂 - 配体组合推荐'
    )
    parser.add_argument(
        '--iteration', '-i',
        action='store_true',
        help='启用迭代模式'
    )
    parser.add_argument(
        '--new-data', '-n',
        type=str,
        help='新实验数据文件路径（Excel 格式）'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        iteration_mode=args.iteration,
        new_data_path=args.new_data
    )
