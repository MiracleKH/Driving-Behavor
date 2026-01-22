import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from tensorflow.keras.optimizers import Adam

# ======================================
# 全局设置：LSTM 序列长度
# ======================================
TIME_STEPS = 10
# ======================================
# 特征提取函数
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


# ======================================
# 数据读取
# ======================================
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


# ======================================
# 序列数据转换函数
# ======================================
def create_sequences(X, y, time_steps):
    """
    将平面特征数据 X 和标签 y 转换为 LSTM 所需的 3D 序列数据。
    """
    X_seq, y_seq = [], []
    X_array = X.values  # X (Pandas DataFrame) 转换为 numpy 数组
    y_array = y  # y (NumPy Array) 直接使用

    for i in range(len(X) - time_steps + 1):
        # 序列从 i 到 i + time_steps
        X_seq.append(X_array[i:(i + time_steps)])
        # 标签为序列的最后一个样本的标签 (i + time_steps - 1)
        y_seq.append(y_array[i + time_steps - 1])

    return np.array(X_seq), np.array(y_seq)


# ======================================
# 随机森林模型训练函数
# ======================================
def train_component_model(df, name, saving_dir):
    X = df.drop(columns=["label", "channel"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n===== [RF] {name} 分类报告 =====")
    print(classification_report(y_test, y_pred))

    # 计算混淆矩阵
    all_labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    # 转换为百分比形式 (按行归一化：即每个真实标签的预测分布)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent_df = pd.DataFrame(cm_normalized * 100,
                                 index=all_labels,
                                 columns=all_labels)

    # 绘制百分比热图
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_percent_df, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f"[RF] {name} Normalized Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 保存百分比 CM 到 CSV
    cm_percent_df.to_csv(os.path.join(saving_dir, f"{name}_RF_CM_percent.csv"))
    print(f" {name}_RF_CM_percent.csv (百分比) 已保存至 /{saving_dir}")

    # 保存模型
    joblib.dump(model, os.path.join(saving_dir, f"{name}_RF_model.pkl"))
    print(f" {name}_RF_model.pkl 已保存至 /{saving_dir}")
    return model


# ======================================
# LSTM 模型训练函数
# ======================================
def train_lstm_model(df, name, time_steps, saving_dir):
    X = df.drop(columns=["label", "channel"])
    y = df["label"]

    print(f" [LSTM] **{name}** 数据集中的所有标签: {sorted(y.unique())}")

    # 1. 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 2. 特征标准化 (在转换为序列前进行)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 3. 转换为 3D 序列数据
    X_seq, y_seq = create_sequences(X_scaled_df, y_encoded, time_steps)
    y_categorical = to_categorical(y_seq)

    # 检查是否有足够的序列数据
    if X_seq.shape[0] < 50:
        print(f" 序列数量不足 ({X_seq.shape[0]})，LSTM 训练可能不稳定。考虑减少 TIME_STEPS 或增加数据。")

    # 划分训练集和测试集 (注意 stratify 使用 y_seq)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_categorical, test_size=0.3, stratify=y_seq, random_state=42
    )

    n_timesteps = X_train.shape[1]  # TIME_STEPS
    n_features = X_train.shape[2]  # 特征数 (13)
    n_classes = y_categorical.shape[1]

    # 4. 构建 LSTM 模型
    print(f" 采用 LSTM 模型，输入形状: ({n_timesteps}, {n_features})")
    model = Sequential([
        # LSTM 层需要 3D 输入 (timesteps, features)
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. 训练模型
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # 6. 绘制历史图并保存损失函数历史到 CSV
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history_df.index + 1  # epoch从1开始

    # 保存损失历史到 CSV
    history_df[['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']].to_csv(
        os.path.join(saving_dir, f"{name}_LSTM_loss_history.csv"), index=False)
    print(f" {name}_LSTM_loss_history.csv (损失函数历史) 已保存至 /{saving_dir}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'[LSTM] {name} Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'[LSTM] {name} Accuracy History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 7. 评估和报告
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n===== [LSTM] {name} 最终测试集性能 =====")
    print(f"  Loss (损失函数值): {loss:.4f}")
    print(f"  Accuracy (准确率): {acc:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)
    y_test_encoded = np.argmax(y_test, axis=1)

    # 还原为原始标签进行报告和混淆矩阵绘制
    y_test_labels = le.inverse_transform(y_test_encoded)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    print(f"\n===== [LSTM] {name} 分类报告 =====")
    print(classification_report(y_test_labels, y_pred_labels))

    # 计算混淆矩阵
    all_labels = sorted(le.classes_.tolist())
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=all_labels)

    # 转换为百分比形式 (按行归一化：即每个真实标签的预测分布)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent_df = pd.DataFrame(cm_normalized * 100,
                                 index=all_labels,
                                 columns=all_labels)

    # 绘制百分比热图
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_percent_df, annot=True, fmt='.2f', cmap='Purples',
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f"[LSTM] {name} Normalized Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 保存百分比 CM 到 CSV
    cm_percent_df.to_csv(os.path.join(saving_dir, f"{name}_LSTM_CM_percent.csv"))
    print(f" {name}_LSTM_CM_percent.csv (百分比) 已保存至 /{saving_dir}")

    # 保存模型、Scaler和LabelEncoder
    model.save(os.path.join(saving_dir, f"{name}_LSTM_model.h5"))
    joblib.dump(scaler, os.path.join(saving_dir, f"{name}_LSTM_scaler.pkl"))
    joblib.dump(le, os.path.join(saving_dir, f"{name}_LSTM_le.pkl"))
    print(f" {name}_LSTM_model.h5, scaler, le 已保存至 /{saving_dir}")

    # 8. 定义 LSTMWrapper 用于后续预测
    class LSTMWrapper:
        def __init__(self, model, scaler, le, time_steps):
            self.model = model
            self.scaler = scaler
            self.le = le
            self.time_steps = time_steps

        def predict(self, X_df):
            X_scaled = self.scaler.transform(X_df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

            # 创建序列，忽略标签 (y 填充为 0)
            X_seq, _ = create_sequences(
                X_scaled_df,
                pd.Series([0] * len(X_scaled_df), index=X_scaled_df.index),  # 确保 y 的长度和索引匹配
                self.time_steps
            )

            # 检查是否有足够的序列数据进行预测
            if len(X_seq) == 0:
                # 如果数据不足以形成一个序列，全部预测为 'Normal' 或抛出错误
                return np.array(["Normal"] * len(X_df))

            # 进行预测
            y_pred_probs = self.model.predict(X_seq, verbose=0)
            y_pred_encoded = np.argmax(y_pred_probs, axis=1)

            # LSTM 的输出长度比原输入短 time_steps - 1，需要进行填充
            initial_preds_count = len(X_df) - len(y_pred_encoded)
            # 简单填充：假设前 initial_preds_count 个样本为 'Normal'
            initial_preds = ["Normal"] * initial_preds_count

            # 还原标签
            final_preds = self.le.inverse_transform(y_pred_encoded)

            # 组合填充值和预测值
            return np.array(initial_preds + final_preds.tolist())

    return LSTMWrapper(model, scaler, le, time_steps)


# ======================================
# 主程序
# ======================================
if __name__ == "__main__":
    ROOT = r"G:\pythonfiles\Driving Behavior\pythonProject\Data"


    SAVING_DIR = "saving"
    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)

    print(" 开始读取四个部件数据...")

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

    models = {}
    component_preds = {}

    # 逐个部件训练
    for component, structure in STRUCTURES.items():
        folder = os.path.join(ROOT, component)
        name = component.replace(" ", "_")
        print(f"\n 正在处理 {component} ...")

        df = load_component_data(folder, structure)
        print(f" {component} 数据读取完成，共 {df.shape[0]} 条样本")

        #  逻辑判断：Seat cushion 使用 LSTM，其他使用 RF
        if component == "Seat cushion":
            print(f" 为 {component} 采用**LSTM (循环神经网络)** 模型 (TIME_STEPS={TIME_STEPS})。")
            # 调用 LSTM 训练函数，并传入 TIME_STEPS 和 SAVING_DIR
            model = train_lstm_model(df, name, TIME_STEPS, SAVING_DIR)
            # LSTMWrapper.predict 接收 DataFrame
            df["pred"] = model.predict(df.drop(columns=["label", "channel"]))
        else:
            print(f" 为 {component} 采用**随机森林 (RF)** 模型。")
            # 调用 RF 训练函数，并传入 SAVING_DIR
            model = train_component_model(df, name, SAVING_DIR)
            df["pred"] = model.predict(df.drop(columns=["label", "channel"]))

        models[component] = model
        component_preds[component] = df

    # ======================================
    # 多部件结果融合
    # ======================================

    min_len = min([len(df) for df in component_preds.values()])
    true_combined, pred_combined = [], []

    for i in range(min_len):

        t_list, p_list = [], []

        # 安全带
        sb_true = component_preds["Seat belt"].iloc[i]["label"]
        sb_pred = component_preds["Seat belt"].iloc[i]["pred"]
        if sb_true != "Normal": t_list.append(sb_true)
        if sb_pred != "Normal": p_list.append(sb_pred)

        # 方向盘
        sw_true = component_preds["Steering wheel"].iloc[min(i, len(component_preds["Steering wheel"]) - 1)]["label"]
        sw_pred = component_preds["Steering wheel"].iloc[min(i, len(component_preds["Steering wheel"]) - 1)]["pred"]
        if sw_true != "Normal": t_list.append(sw_true)
        if sw_pred != "Normal": p_list.append(sw_pred)

        # 踏板
        pd_true = component_preds["Pedal"].iloc[min(i, len(component_preds["Pedal"]) - 1)]["label"]
        pd_pred = component_preds["Pedal"].iloc[min(i, len(component_preds["Pedal"]) - 1)]["pred"]
        if pd_true != "Normal": t_list.append(pd_true)
        if pd_pred != "Normal": p_list.append(pd_pred)

        # 坐垫
        sc_true = component_preds["Seat cushion"].iloc[min(i, len(component_preds["Seat cushion"]) - 1)]["label"]
        sc_pred = component_preds["Seat cushion"].iloc[min(i, len(component_preds["Seat cushion"]) - 1)]["pred"]
        if sc_true != "Normal": t_list.append(sc_true)
        if sc_pred != "Normal": p_list.append(sc_pred)

