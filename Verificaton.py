import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ======================================
# 验证脚本配置
# ======================================
# ⚠️ 1. 请修改为您的新数据根目录
NEW_DATA_ROOT = r"G:\pythonfiles\Driving Behavior\pythonProject\Verification_Data"
# ⚠️ 2. 请确保 SAVING_DIR 与训练脚本中的一致
SAVING_DIR = "saving"
# 3. 序列长度必须与训练时一致
TIME_STEPS = 10

# 定义用于保存结果的目录
# CSV 文件将保存到 NEW_DATA_ROOT/Verification_Results 目录下
VERIFICATION_RESULTS_DIR = os.path.join(NEW_DATA_ROOT, "Verification_Results")

# ======================================
# 部件结构（必须与训练时一致）
# ======================================
STRUCTURES = {
    "Seat belt": {"Belly": ["Crush", "Normal"], "Chest": ["Crush", "Normal"]},
    "Steering wheel": {"3": ["Grip heavily", "Leave with both hands", "Normal"],
                       "9": ["Grip heavily", "Leave with both hands", "Normal"]},
    "Pedal": {"": ["Step", "Normal"]},
    "Seat cushion": {"1.1": ["Back", "Forward", "Left", "Right", "Normal"],
                     "1.2": ["Back", "Forward", "Left", "Right", "Normal"],
                     "2.1": ["Back", "Forward", "Left", "Right", "Normal"],
                     "2.2": ["Back", "Forward", "Left", "Right", "Normal"]}
}


# ======================================
# 辅助函数（来自原训练脚本）
# ======================================

def extract_features_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=8, header=None)
        df.columns = ['time', 'cap']
        if len(df) < 10:
            return None
    except Exception:
        return None

    x = df['cap'].values
    feats = {
        'mean': np.mean(x),
        'std': np.std(x),
        'min': np.min(x),
        'max': np.max(x),
        'range': np.ptp(x),
        'skew': skew(x),
        'kurtosis': kurtosis(x),
        'energy': np.sum(x ** 2) / len(x),
        'median': np.median(x),
        'iqr': np.percentile(x, 75) - np.percentile(x, 25)
    }

    if len(df) > 1:
        dt = np.mean(np.diff(df['time'].values))
        deriv = np.gradient(x)
        feats['sampling_interval'] = dt
        feats['derivative_mean'] = np.mean(deriv)
        feats['derivative_std'] = np.std(deriv)
    else:
        feats['sampling_interval'] = 0
        feats['derivative_mean'] = 0
        feats['derivative_std'] = 0

    return feats


def load_component_data(root_folder, substructure):
    data, labels = [], []
    for channel, states in substructure.items():
        channel_path = os.path.join(root_folder, channel)
        if not os.path.isdir(channel_path) and channel != "":
            continue
        for state in states:
            if channel == "":
                state_path = os.path.join(root_folder, state)
            else:
                state_path = os.path.join(channel_path, state)
            if not os.path.isdir(state_path):
                continue
            for file in os.listdir(state_path):
                if not file.endswith(".csv"):
                    continue
                feats = extract_features_from_csv(os.path.join(state_path, file))
                if feats is not None:
                    feats["channel"] = channel
                    data.append(feats)
                    labels.append(state)
    df = pd.DataFrame(data)
    df["label"] = labels
    return df


def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    X_array = X.values
    y_array = y

    for i in range(len(X) - time_steps + 1):
        X_seq.append(X_array[i:(i + time_steps)])
        y_seq.append(y_array[i + time_steps - 1])

    return np.array(X_seq), np.array(y_seq)


class LSTMWrapper:
    def __init__(self, model, scaler, le, time_steps):
        self.model = model
        self.scaler = scaler
        self.le = le
        self.time_steps = time_steps

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

        X_seq, _ = create_sequences(
            X_scaled_df,
            pd.Series([0] * len(X_scaled_df), index=X_scaled_df.index),
            self.time_steps
        )

        if len(X_seq) == 0:
            return np.array(["Normal"] * len(X_df))

        y_pred_probs = self.model.predict(X_seq, verbose=0)
        y_pred_encoded = np.argmax(y_pred_probs, axis=1)

        initial_preds_count = len(X_df) - len(y_pred_encoded)
        initial_preds = ["Normal"] * initial_preds_count

        final_preds = self.le.inverse_transform(y_pred_encoded)

        return np.array(initial_preds + final_preds.tolist())


# ======================================
# 主预测逻辑 (已更新，包含混淆矩阵CSV输出)
# ======================================
def validate_new_data(root_folder, structures, saving_dir, time_steps, results_dir):
    component_preds = {}
    print(f"🚀 开始加载模型和新数据 (ROOT: {root_folder})")

    # 确保结果保存目录存在
    os.makedirs(results_dir, exist_ok=True)
    print(f"💾 结果文件将保存到目录: {results_dir}")

    # 1. 逐个部件加载模型和数据，并进行预测
    for component, structure in structures.items():
        name = component.replace(" ", "_")
        folder = os.path.join(root_folder, component)

        # 1.1 加载新数据
        df_new = load_component_data(folder, structure)

        # 提取特征数据，用于预测
        X_new = df_new.drop(columns=["label", "channel"])
        y_true = df_new["label"].values  # 真实标签用于评估

        print(f"\n📂 正在处理 {component}，样本数: {df_new.shape[0]}")

        if df_new.empty:
            print(f"⚠️ {component} 数据为空，跳过。")
            continue

        # 1.2 加载模型文件
        model_type = "RF"
        y_pred = None

        if component == "Seat cushion":
            # LSTM 模型
            model_path = os.path.join(saving_dir, f"{name}_LSTM_model.h5")
            scaler_path = os.path.join(saving_dir, f"{name}_LSTM_scaler.pkl")
            le_path = os.path.join(saving_dir, f"{name}_LSTM_le.pkl")

            if not os.path.exists(model_path):
                print(f"❌ 找不到 LSTM 模型文件: {model_path}，跳过。")
                continue

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            le = joblib.load(le_path)

            # 使用 LSTMWrapper 进行预测
            lstm_wrapper = LSTMWrapper(model, scaler, le, time_steps)
            y_pred = lstm_wrapper.predict(X_new)
            model_type = "LSTM"

        else:
            # RF 模型
            model_path = os.path.join(saving_dir, f"{name}_RF_model.pkl")
            if not os.path.exists(model_path):
                print(f"❌ 找不到 RF 模型文件: {model_path}，跳过。")
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(X_new)

        # 1.3 存储预测结果
        df_new["pred"] = y_pred
        component_preds[component] = df_new

        # --- 保存Seat Cushion的预测结果到CSV (用于诊断) ---
        if component == "Seat cushion":
            output_path = os.path.join(results_dir, f"{name}_raw_predictions.csv")
            df_new[['label', 'pred']].to_csv(output_path, index=False)
            print(f"📄 已将 {component} 的原始预测结果保存到: {output_path}")
        # -------------------------------------------------------------------

        print(f"✅ {component} 预测完成 ({model_type})")

        # 1.4 报告该部件的性能
        print(f"\n--- {component} 独立报告 (新数据) ---")
        print(classification_report(y_true, y_pred, zero_division=0))

        # 1.5 计算并保存混淆矩阵

        # 提取所有可能标签并排序
        all_possible_labels_set = set()
        for states in structure.values():
            all_possible_labels_set.update(states)
        all_labels = sorted(list(all_possible_labels_set))

        # 计算混淆矩阵 (使用所有可能的标签作为维度)
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        # 转换为百分比形式 (按行归一化)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm.astype('float'),
                                  row_sums,
                                  out=np.zeros_like(cm.astype('float')),
                                  where=row_sums != 0)

        cm_percent_df = pd.DataFrame(cm_normalized * 100,
                                     index=[f"True_{lab}" for lab in all_labels], # 明确行是真实标签
                                     columns=[f"Pred_{lab}" for lab in all_labels]) # 明确列是预测标签

        # 保存混淆矩阵数据到 CSV
        cm_output_path = os.path.join(results_dir, f"{name}_normalized_cm.csv")
        cm_percent_df.to_csv(cm_output_path)
        print(f"💾 已将 {component} 归一化混淆矩阵数据保存到: {cm_output_path}")


        print(f"\n📢 {component} 独立混淆矩阵 (行归一化百分比):")
        # 输出到控制台
        print(cm_percent_df.to_string())

        # 绘制混淆矩阵热图
        plt.figure(figsize=(7, 6))
        cmap_style = 'Blues' if model_type == 'RF' else 'Purples'
        sns.heatmap(cm_percent_df.values, annot=True, fmt='.2f', cmap=cmap_style,
                    xticklabels=all_labels, yticklabels=all_labels)
        plt.title(f"[New Data] {component} Normalized Confusion Matrix (%)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # 2. 融合预测结果 (保持不变)
    if not component_preds:
        print("\n无法进行融合：没有部件数据或模型加载失败。")
        return

    print("\n🔗 开始融合预测结果...")

    min_len = min([len(df) for df in component_preds.values()])
    true_combined, pred_combined = [], []

    for i in range(min_len):
        t_list, p_list = [], []

        # 遍历所有部件，收集真实标签和预测标签
        for component in structures.keys():
            if component in component_preds:
                df = component_preds[component]
                idx = min(i, len(df) - 1)

                true_label = df.iloc[idx]["label"]
                pred_label = df.iloc[idx]["pred"]

                if true_label != "Normal": t_list.append(true_label)
                if pred_label != "Normal": p_list.append(pred_label)

        true_state = "+".join(sorted(t_list)) if t_list else "Normal"
        pred_state = "+".join(sorted(p_list)) if p_list else "Normal"

        true_combined.append(true_state)
        pred_combined.append(pred_state)

    # 3. 输出融合报告 (包含最终混淆矩阵CSV输出)
    all_combined_labels = sorted(list(set(true_combined + pred_combined)))

    print("\n\n#####################################################")
    print("      ✨ 新数据：多部件融合分类报告 (LSTM + RF) ✨         ")
    print("#####################################################")
    print(classification_report(true_combined, pred_combined, zero_division=0))

    cm_final = confusion_matrix(true_combined, pred_combined, labels=all_combined_labels)

    row_sums = cm_final.sum(axis=1)[:, np.newaxis]
    cm_final_normalized = np.divide(cm_final.astype('float'),
                                    row_sums,
                                    out=np.zeros_like(cm_final.astype('float')),
                                    where=row_sums != 0)

    cm_final_percent_df = pd.DataFrame(cm_final_normalized * 100,
                                       index=[f"True_{lab}" for lab in all_combined_labels],
                                       columns=[f"Pred_{lab}" for lab in all_combined_labels])

    # 保存最终融合的混淆矩阵数据到 CSV
    final_cm_output_path = os.path.join(results_dir, "Combined_final_normalized_cm.csv")
    cm_final_percent_df.to_csv(final_cm_output_path)
    print(f"💾 已将 **最终融合** 归一化混淆矩阵数据保存到: {final_cm_output_path}")


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_final_percent_df.values, annot=True, fmt='.2f', cmap='magma',
                xticklabels=all_combined_labels, yticklabels=all_combined_labels)
    plt.title("New Data: Final Combined Normalized Confusion Matrix (%)")
    plt.xlabel("Predicted Combined State")
    plt.ylabel("True Combined State")
    plt.show()


if __name__ == "__main__":
    validate_new_data(NEW_DATA_ROOT, STRUCTURES, SAVING_DIR, TIME_STEPS, VERIFICATION_RESULTS_DIR) 
