import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import tkinter as tk
from tkinter import ttk, filedialog
from functools import partial


try:
    from tensorflow.keras.models import load_model
except ImportError:

    def load_model(path):
        raise ImportError("TensorFlow/Keras is required for LSTM models.")

# ======================================
# 全局设置
# ======================================
TIME_STEPS = 10
LOADED_MODELS = None
LOADED_CMS = None
NEW_DATA_ROOT = None
ALL_ACTION_GROUPS = []


# ======================================
# 原始数据处理函数
# ======================================
def extract_features_from_csv(file_path):
    """从 CSV 文件中提取特征。"""
    try:
        # skiprows=8 确保跳过前 8 行测试信息
        df = pd.read_csv(file_path, skiprows=8, header=None)
        df.columns = ['time', 'cap']

        if len(df) < 1:
            return None

    except Exception:
        return None

    x = df['cap'].values
    # 提取统计特征
    feats = {
        'mean': np.mean(x), 'std': np.std(x), 'min': np.min(x),
        'max': np.max(x), 'range': np.ptp(x), 'skew': skew(x),
        'kurtosis': kurtosis(x), 'energy': np.sum(x ** 2) / len(x),
        'median': np.median(x), 'iqr': np.percentile(x, 75) - np.percentile(x, 25)
    }

    # 提取微分特征
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


def create_sequences(X, y, time_steps):
    """将平面特征数据 X 和标签 y 转换为 LSTM 所需的 3D 序列数据。"""
    X_seq, y_seq = [], []
    X_array = X.values
    y_array = y

    for i in range(len(X) - time_steps + 1):
        X_seq.append(X_array[i:(i + time_steps)])
        y_seq.append(y_array[i + time_steps - 1])

    return np.array(X_seq), np.array(y_seq)


# ======================================
# 模型预测器封装类
# ======================================

class LSTMPredictor:
    def __init__(self, model_path, scaler_path, le_path, time_steps):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.le = joblib.load(le_path)
        self.time_steps = time_steps
        self.feature_names = self.scaler.feature_names_in_

    def predict_proba(self, X_df):
        X_df = X_df[self.feature_names]
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

        y_placeholder = pd.Series([0] * len(X_scaled_df), index=X_scaled_df.index)
        X_seq, _ = create_sequences(X_scaled_df, y_placeholder, self.time_steps)

        n_classes = len(self.le.classes_)
        full_probs = np.zeros((len(X_df), n_classes))

        if len(X_seq) == 0: return full_probs

        y_pred_probs_seq = self.model.predict(X_seq, verbose=0)

        # 处理序列模型前 time_steps - 1 个样本的 Normal 填充
        if 'Normal' in self.le.classes_:
            normal_idx = np.where(self.le.classes_ == 'Normal')[0][0]
            for i in range(min(self.time_steps - 1, len(X_df))):
                full_probs[i, normal_idx] = 1.0

        start_idx = self.time_steps - 1
        end_idx = start_idx + len(y_pred_probs_seq)
        full_probs[start_idx:end_idx] = y_pred_probs_seq

        return full_probs


class RFPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.le_classes = self.model.classes_
        self.feature_names = self.model.feature_names_in_

    def predict_proba(self, X_df):
        X_df = X_df[self.feature_names]
        return self.model.predict_proba(X_df)


# ======================================
# 模型加载和性能函数
# ======================================
def load_all_models(saving_dir="saving"):
    """从保存目录加载所有部件的预测模型和训练时的归一化混淆矩阵。"""
    MODELS = {}
    CMS = {}
    STRUCTURES = {
        "Seat belt": "RF",
        "Steering wheel": "RF",
        "Pedal": "RF",
        "Seat cushion": "LSTM"
    }

    if not os.path.isdir(saving_dir):
        raise FileNotFoundError(f"找不到模型保存目录: '{saving_dir}'。请确保模型已训练并保存在此。")

    for component, model_type in STRUCTURES.items():
        name = component.replace(" ", "_")
        cm_filename = f"{name}_{model_type}_CM_percent.csv"
        cm_path = os.path.join(saving_dir, cm_filename)

        try:
            # 1. 加载模型
            if model_type == "RF":
                model_path = os.path.join(saving_dir, f"{name}_RF_model.pkl")
                MODELS[component] = RFPredictor(model_path)
            elif model_type == "LSTM":
                model_path = os.path.join(saving_dir, f"{name}_LSTM_model.h5")
                scaler_path = os.path.join(saving_dir, f"{name}_LSTM_scaler.pkl")
                le_path = os.path.join(saving_dir, f"{name}_LSTM_le.pkl")
                MODELS[component] = LSTMPredictor(model_path, scaler_path, le_path, TIME_STEPS)

            # 2. 加载混淆矩阵
            if os.path.exists(cm_path):
                cm_df = pd.read_csv(cm_path, index_col=0)
                cm_text = f"\n=== {component} (训练集CM %) ===\n"
                cm_text += cm_df.to_string(
                    float_format=lambda x: f'{x:.2f}' if isinstance(x, (float, np.floating)) else str(x))
                CMS[component] = cm_text
            else:
                CMS[component] = f"\n=== {component} (训练集CM %) ===\n 找不到混淆矩阵文件: {cm_filename}"

        except Exception as e:
            raise Exception(f"加载 {component} 模型/CM失败: {e}")

    return MODELS, CMS


def apply_threshold_and_get_state(results, threshold=0.6):
    """
    只要任一非 Normal 状态概率超过阈值，则采用概率最高的非 Normal 状态作为最终结果。
    """
    final_states = {}
    for file_name, component_results in results.items():
        final_states[file_name] = {}
        for component, state_probs in component_results.items():
            final_state = "Normal"  # 默认状态
            max_prob = 0.0
            best_non_normal_state = "Normal"

            # 第一次遍历：找到概率最高的非 Normal 状态及其概率
            for state, prob in state_probs.items():
                if state != "Normal":
                    if prob > max_prob:
                        max_prob = prob
                        best_non_normal_state = state

            # 第二次判断：如果最高的非 Normal 概率达到阈值 (60% 或更高)，则采用该状态
            if max_prob >= threshold:
                final_state = best_non_normal_state

            # 否则，保持默认的 "Normal"
            final_states[file_name][component] = final_state
    return final_states
# ======================================
# 数据加载和预测主函数
# ======================================
def scan_action_groups(root_path):
    """扫描指定根目录，返回所有唯一的 CSV 文件名（动作组 ID）"""
    NEW_STRUCTURE_COMPONENTS = ["Seat belt", "Pedal", "Seat cushion", "Steering wheel"]
    unique_files = set()

    for component in NEW_STRUCTURE_COMPONENTS:
        comp_path = os.path.join(root_path, component)
        if not os.path.isdir(comp_path): continue
        for dirpath, dirnames, filenames in os.walk(comp_path):
            for file in filenames:
                if file.endswith(".csv"):
                    unique_files.add(file)

    return sorted(list(unique_files))


def load_new_data_and_predict(new_data_root, models, group_filter=None):
    """加载数据并预测，返回平均概率和样本预测计数。"""

    NEW_STRUCTURE = {
        "Seat belt": ["Belly", "Chest"],
        "Pedal": [""],
        "Seat cushion": ["1.1", "1.2", "2.1", "2.2"],
        "Steering wheel": ["3", "9"]
    }

    action_files = {}

    # 阶段 1: 特征提取和按动作分组
    for component, channels in NEW_STRUCTURE.items():
        comp_path = os.path.join(new_data_root, component)
        if not os.path.isdir(comp_path): continue
        for channel in channels:
            dir_path = comp_path if channel == "" else os.path.join(comp_path, channel)
            if not os.path.isdir(dir_path): continue
            for file in os.listdir(dir_path):
                if not file.endswith(".csv"): continue
                if group_filter is not None and file != group_filter: continue
                full_path = os.path.join(dir_path, file)
                feats = extract_features_from_csv(full_path)
                if feats is None: continue
                feats_df = pd.DataFrame([feats])
                if file not in action_files:
                    action_files[file] = {comp: {} for comp in NEW_STRUCTURE.keys()}
                action_files[file][component][channel] = feats_df

    # 阶段 2: 概率预测和计数
    action_results = {}
    sample_pred_counts = {}

    if not action_files:
        raise ValueError("在新数据根目录中未找到任何有效的 CSV 文件进行处理。")

    for file_name, component_data in tqdm(action_files.items(), desc=" 正在对动作组进行预测"):

        action_results[file_name] = {}
        sample_pred_counts[file_name] = {}

        for component, channel_data in component_data.items():

            if not channel_data: continue

            full_feature_df = pd.concat(channel_data.values(), ignore_index=True)
            model_predictor = models[component]

            # 1. 进行概率预测
            probs_matrix = model_predictor.predict_proba(full_feature_df)

            # 2. 确定样本预测的最终类别 (取概率最高的)
            predicted_encoded = np.argmax(probs_matrix, axis=1)

            # 3. 还原为原始标签
            classes = model_predictor.le.classes_.tolist() if hasattr(model_predictor,
                                                                      'le') else model_predictor.le_classes.tolist()
            predicted_labels = [classes[i] for i in predicted_encoded]

            # 4. 统计计数
            counts = pd.Series(predicted_labels).value_counts().to_dict()
            counts['__Total__'] = len(predicted_labels)
            sample_pred_counts[file_name][component] = counts

            # 5. 计算部件的平均概率 (用于 60% 阈值判断)
            avg_probs = np.mean(probs_matrix, axis=0)
            result = {classes[i]: avg_probs[i] for i in range(len(classes))}
            action_results[file_name][component] = result

    return action_results, sample_pred_counts


# ======================================
# GUI 回调函数
# ======================================
def load_models_for_gui(status_label, loading_button, cm_text_widget):
    """在 GUI 中加载所有模型和混淆矩阵"""
    global LOADED_MODELS, LOADED_CMS
    try:
        status_label.config(text="正在加载模型和混淆矩阵... 请稍候...", foreground="orange")
        status_label.winfo_toplevel().update()

        MODELS, CMS = load_all_models(saving_dir="saving")
        globals()['LOADED_MODELS'] = MODELS
        globals()['LOADED_CMS'] = CMS

        # 显示混淆矩阵
        cm_text_widget.delete('1.0', tk.END)
        full_cm_text = "\n".join(CMS.values())
        cm_text_widget.insert(tk.END, full_cm_text)

        status_label.config(text="模型和混淆矩阵加载成功！", foreground="green")
        loading_button.config(state=tk.DISABLED)
    except Exception as e:
        status_label.config(text=f"加载失败: {e}", foreground="red")
        globals()['LOADED_MODELS'] = None
        globals()['LOADED_CMS'] = None


def select_folder_and_scan(status_label, select_group_combobox, run_btn):
    """选择文件夹并扫描动作组"""
    global NEW_DATA_ROOT, ALL_ACTION_GROUPS

    if LOADED_MODELS is None:
        status_label.config(text="请先加载模型！", foreground="red")
        return

    new_data_root = filedialog.askdirectory(title="选择新数据根目录")
    if not new_data_root:
        return

    globals()['NEW_DATA_ROOT'] = new_data_root

    try:
        status_label.config(text="正在扫描动作组...", foreground="blue")
        status_label.winfo_toplevel().update()

        globals()['ALL_ACTION_GROUPS'] = scan_action_groups(NEW_DATA_ROOT)

        if not ALL_ACTION_GROUPS:
            status_label.config(text="未找到任何有效的 CSV 动作组。", foreground="red")
            select_group_combobox.set("")
            run_btn.config(state=tk.DISABLED)
            return

        status_label.config(
            text=f"已找到 {len(ALL_ACTION_GROUPS)} 个动作组。请在下拉列表中选择。",
            foreground="green"
        )

    except Exception as e:
        status_label.config(text=f"扫描错误: {e}", foreground="red")
        run_btn.config(state=tk.DISABLED)


def run_prediction(text_widget, status_label, selected_group_var):
    """运行所选动作组的预测，并显示文本报告。"""
    global LOADED_MODELS, NEW_DATA_ROOT

    group_filter = selected_group_var.get()
    if group_filter == "":
        status_label.config(text="请选择一个动作组！", foreground="red")
        return

    status_label.config(text=f"正在预测动作组: {group_filter}...", foreground="blue")
    text_widget.delete('1.0', tk.END)
    status_label.winfo_toplevel().update()

    try:
        # 1. 加载数据并预测概率
        results_avg_prob, results_counts = load_new_data_and_predict(
            NEW_DATA_ROOT, LOADED_MODELS, group_filter=group_filter
        )

        file_name = list(results_counts.keys())[0]
        counts_data = results_counts[file_name]

        # 2. 直接根据样本预测统计结果确定最终状态
        final_states = {file_name: {}}
        for component in counts_data:
            counts = counts_data[component]
            total = counts.get('__Total__', 0)

            if total == 0:
                final_states[file_name][component] = "Normal"
                continue

            # 找到非 Normal 状态中样本数最高的
            max_non_normal_count = -1
            best_state = "Normal"

            for state, count in counts.items():
                if state == '__Total__': continue

                if state != "Normal":
                    if count > max_non_normal_count:
                        max_non_normal_count = count
                        best_state = state
                elif count == total and total > 0:
                    # 如果所有样本都是 Normal，则确定为 Normal
                    best_state = "Normal"

            # 最终判定：如果最高的非 Normal 状态样本数大于等于 60%
            threshold_count = total * 0.6
            if max_non_normal_count >= threshold_count:
                # 采纳样本数最高的非 Normal 状态
                final_states[file_name][component] = best_state
            else:
                # 否则，判断为 Normal (即使 Normal 样本数不足 60%)
                final_states[file_name][component] = "Normal"

        states = final_states[file_name]

        # 3. 格式化输出
        output_text = f"========== 动作组: {file_name} ==========\n\n"
        output_text += "--- [A] 样本统计阈值判定结果 (最终判断) ---\n"

        # 插入纯文本部分
        text_widget.insert(tk.END, output_text)

        for component in sorted(states.keys()):
            state = states[component]

            # 格式化组件名称和前导符
            prefix = "  - "
            component_name_part = f"{component:<15}:"

            # 确定显示文本和要应用的 Tag
            if state == "Normal":
                display_state_part = "😁😀 正常\n"
                tag_name = 'normal_tag'  # 绿色加粗
            else:
                # 统一将所有非 Normal 状态识别为姿态异常，并显示具体状态
                display_state_part = f"!!⚠️ 姿态异常 ({state}) !!\n"
                tag_name = 'abnormal_tag'  # 红色加粗

            # 按顺序插入文本和应用 Tag
            text_widget.insert(tk.END, prefix)
            text_widget.insert(tk.END, component_name_part, ('normal_tag', 'abnormal_tag'))  # 保持组件名称颜色中立或使用默认
            text_widget.insert(tk.END, display_state_part, tag_name)  # 应用目标颜色 Tag

        # 4. 格式化输出 (样本分布统计)
        output_text_b = "\n--- [B] 样本预测分布统计 (预测倾向) ---\n"
        text_widget.insert(tk.END, output_text_b)

        for component in sorted(counts_data.keys()):
            counts = counts_data[component]
            total = counts.pop('__Total__', 0)

            output_text_comp = f"\n> {component} (总样本数: {total})\n"
            text_widget.insert(tk.END, output_text_comp)

            if total > 0:
                sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

                for state, count in sorted_counts:
                    percentage = (count / total) * 100
                    output_line = f"  {state:<15} | {count:>5} 样本 | {percentage:>6.2f}%\n"
                    # 这里可以根据需要，只对非 Normal 且占比高的行应用红色 tag
                    if state != "Normal" and percentage >= 60.0:
                        text_widget.insert(tk.END, output_line, 'normal_tag')
                    else:
                        text_widget.insert(tk.END, output_line)
            else:
                text_widget.insert(tk.END, "  (无有效样本)\n")

        output_text_end = "\n=========================================="
        text_widget.insert(tk.END, output_text_end)

        status_label.config(text=f"动作组 {group_filter} 预测完成。", foreground="green")

    except Exception as e:
        error_msg = f"预测过程中出现错误: {type(e).__name__}: {e}"
        status_label.config(text=error_msg, foreground="red")
        text_widget.insert(tk.END, f"\n[ERROR] 错误详情:\n{error_msg}")

# ======================================
# GUI 结构定义
# ======================================
def setup_gui():
    """设置主 GUI 界面"""
    root = tk.Tk()
    root.title("驾驶姿势识别验证工具 ")
    root.geometry("800x650")  # 缩小窗口尺寸，因为没有图表

    style = ttk.Style()
    style.configure("TButton", padding=6, font=('Segoe UI', 10, 'bold'))
    style.configure("TLabel", font=('Segoe UI', 10))

    # --- 主框架 ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill='both', expand=True)

    # --- 顶部操作区 ---
    top_panel = ttk.Frame(main_frame)
    top_panel.pack(fill='x')

    status_label = ttk.Label(top_panel, text="欢迎使用！请先加载模型。", foreground="gray")
    status_label.pack(side=tk.LEFT, padx=5, pady=5)

    # --- 中间：结果/CM 显示区 (Notebook) ---
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill='both', expand=True, pady=10)

    # Tab 1: 混淆矩阵 (CM) 页面
    cm_frame = ttk.Frame(notebook, padding="10")
    notebook.add(cm_frame, text="模型训练性能 (CM)")
    cm_text_widget = tk.Text(cm_frame, wrap=tk.WORD, font=("Consolas", 9), bd=2, relief=tk.SUNKEN)
    cm_text_widget.pack(fill='both', expand=True)

    # Tab 2: 预测结果页面 (单个文本框)
    result_frame = ttk.Frame(notebook, padding="10")
    notebook.add(result_frame, text="新数据预测结果 ")

    # 单个文本报告区域
    text_widget = tk.Text(result_frame, wrap=tk.WORD, font=("Consolas", 10), bd=2, relief=tk.SUNKEN)
    text_widget.pack(fill='both', expand=True, side=tk.LEFT)
    text_widget.tag_configure('abnormal_tag',
                              foreground='red',
                              font=('Consolas', 10, 'bold'))
    # 定义绿色正常字体样式 (可选，但为了对比度更好)
    text_widget.tag_configure('normal_tag',
                              foreground='green',
                              font=('Consolas', 10, 'normal'))
    scrollbar = ttk.Scrollbar(result_frame, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill='y')
    text_widget.config(yscrollcommand=scrollbar.set)

    # --- 底部操作区 ---
    bottom_bar = ttk.Frame(main_frame, padding="5")
    bottom_bar.pack(fill='x')

    # 步骤 3 变量和组件
    ttk.Label(bottom_bar, text="3. 选择动作组:").pack(side=tk.LEFT, padx=(20, 5))
    selected_group_var = tk.StringVar()
    select_group_combobox = ttk.Combobox(
        bottom_bar, textvariable=selected_group_var, state='readonly', width=20)
    select_group_combobox['values'] = ALL_ACTION_GROUPS
    select_group_combobox.pack(side=tk.LEFT, padx=5)

    # 步骤 4 按钮：运行预测 (注意：不再需要 chart_frame 参数)
    run_btn = ttk.Button(
        bottom_bar, text="4. 运行预测", state=tk.DISABLED,
        command=partial(run_prediction, text_widget, status_label, selected_group_var))
    run_btn.pack(side=tk.RIGHT, padx=10)

    # 步骤 1 按钮：加载模型
    load_btn = ttk.Button(top_panel, text="1. 加载模型",
                          command=lambda: load_models_for_gui(status_label, load_btn, cm_text_widget))
    load_btn.pack(side=tk.RIGHT, padx=10, pady=5)

    # 步骤 2 按钮：选择数据文件夹
    def update_combobox_after_scan(combobox, run_button):
        """辅助函数：扫描后更新 ComboBox 的值"""
        combobox['values'] = ALL_ACTION_GROUPS
        if ALL_ACTION_GROUPS:
            combobox.set(ALL_ACTION_GROUPS[0])
            run_button.config(state=tk.NORMAL)
        else:
            combobox.set("")
            run_button.config(state=tk.DISABLED)

    select_data_btn = ttk.Button(
        bottom_bar, text="2. 选择新数据文件夹并扫描动作组",
        command=lambda: [
            select_folder_and_scan(status_label, select_group_combobox, run_btn),
            update_combobox_after_scan(select_group_combobox, run_btn)]
    )
    select_data_btn.pack(side=tk.LEFT, padx=10)

    select_group_combobox.bind("<<ComboboxSelected>>", lambda e: run_btn.config(state=tk.NORMAL))

    root.mainloop()


# ======================================
# 主入口
# ======================================
if __name__ == "__main__":
    print("--- 驾驶姿势识别验证工具已启动 ---")
    setup_gui()