import tkinter as tk
from tkinter import ttk, filedialog

import joblib
from PIL import Image, ImageTk
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tensorflow.keras.models import load_model
import re  # 用于 _find_and_group_data_files

# 模拟行为识别函数
def get_features(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=8, header=None)
        df.columns = ['time', 'cap']
        if len(df) < 10: return None
        x = df['cap'].values
        feats = {
            'mean': np.mean(x), 'std': np.std(x), 'min': np.min(x), 'max': np.max(x),
            'range': np.ptp(x), 'skew': skew(x), 'kurtosis': kurtosis(x),
            'energy': np.sum(x ** 2) / len(x), 'median': np.median(x),
            'iqr': np.percentile(x, 75) - np.percentile(x, 25)
        }
        dt = np.mean(np.diff(df['time'].values)) if len(df) > 1 else 0
        deriv = np.gradient(x) if len(df) > 1 else [0]
        feats['sampling_interval'] = dt
        feats['derivative_mean'] = np.mean(deriv)
        feats['derivative_std'] = np.std(deriv)
        return pd.DataFrame([feats])
    except:
        return None


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Driving Behavior Recognition System")
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry("1200x800")

        self.component_states = {
            "Seat cushion": "N/A",
            "Steering wheel": "N/A",
            "Seat belt": "N/A",
            "Pedal": "N/A"
        }
        self.state_labels = {}
        self.default_image = tk.PhotoImage(width=1, height=1)
        self.sensor_labels = []

        # 用于分组和下拉框
        self.data_groups = {}
        self.selected_group_name = tk.StringVar(value="-- Select Dataset--")
        self.current_filepaths = []
        self.create_widgets()
        #用于加载预训练模型
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.load_all_models()

    def load_all_models(self):
        """从 saving 目录加载所有模型和辅助工具"""
        save_path = "saving"
        try:
            # 加载随机森林模型
            self.models["Seat_belt"] = joblib.load(f"{save_path}/Seat_belt_RF_model.pkl")
            self.models["Steering_wheel"] = joblib.load(f"{save_path}/Steering_wheel_RF_model.pkl")
            self.models["Pedal"] = joblib.load(f"{save_path}/Pedal_RF_model.pkl")

            # 加载 LSTM 相关 (坐垫)
            self.models["Seat_cushion"] = load_model(f"{save_path}/Seat_cushion_LSTM_model.h5")
            self.scalers["Seat_cushion"] = joblib.load(f"{save_path}/Seat_cushion_LSTM_scaler.pkl")
            self.label_encoders["Seat_cushion"] = joblib.load(f"{save_path}/Seat_cushion_LSTM_le.pkl")
            print("所有模型加载成功！")
        except Exception as e:
            print(f"模型加载失败，请检查 /saving 文件夹内容: {e}")

    def start_prediction(self):
        """点击 Prediction 按钮后的逻辑"""
        if not self.current_filepaths:
            print("错误：请先选择数据集")
            return

        # 1. 自动对当前组的文件进行部件分类
        # 这里假设文件名中包含部件关键字（如 'Pedal_1.csv', 'wheel_2.csv'）
        file_mapping = {
            "Pedal": None, "Steering_wheel": None, "Seat_belt": None, "Seat_cushion": []
        }

        for path in self.current_filepaths:
            name = path.lower()
            if "pedal" in name:
                file_mapping["Pedal"] = path
            elif "wheel" in name:
                file_mapping["Steering_wheel"] = path
            elif "belt" in name:
                file_mapping["Seat_belt"] = path
            elif "cushion" in name or "seat" in name:
                file_mapping["Seat_cushion"].append(path)

        # 2. 逐个预测
        results = {"Pedal": "N/A", "Steering_wheel": "N/A", "Seat_belt": "N/A", "Seat_cushion": "N/A"}

        # --- 预测 Pedal, Wheel, Belt (RF模型) ---
        for part in ["Pedal", "Steering_wheel", "Seat_belt"]:
            if file_mapping[part]:
                feat_df = get_features(file_mapping[part])
                if feat_df is not None:
                    results[part] = self.models[part].predict(feat_df)[0]

        # --- 预测 Seat cushion (LSTM模型) ---

        if file_mapping["Seat_cushion"]:
            # 使用其中一个文件提取特征
            feat_df = get_features(file_mapping["Seat_cushion"][0])
            if feat_df is not None:
                scaler = self.scalers["Seat_cushion"]
                le = self.label_encoders["Seat_cushion"]
                X_scaled = scaler.transform(feat_df)
                # 构造符合 LSTM 输入的 3D 形状 (batch, time_steps, features)
                # 假设 TIME_STEPS = 10，我们把当前特征重复10次作为输入
                X_seq = np.repeat(X_scaled[np.newaxis, :, :], 10, axis=1)
                pred_prob = self.models["Seat_cushion"].predict(X_seq, verbose=0)
                results["Seat_cushion"] = le.inverse_transform([np.argmax(pred_prob)])[0]

        self.update_ui_with_logic(results)
    def create_widgets(self):
        # ----------------------------------------------
        # I. 顶部自定义工具栏  - 占据 Row 0
        # ----------------------------------------------

        top_toolbar_frame = ttk.Frame(self, padding="5 5")
        top_toolbar_frame.grid(row=0, column=0, columnspan=3, sticky="new")

        # 配置工具栏的列权重，让下拉框能扩展
        top_toolbar_frame.grid_columnconfigure(0, weight=0)  # 文件按钮
        top_toolbar_frame.grid_columnconfigure(1, weight=0)  # 分隔线
        top_toolbar_frame.grid_columnconfigure(2, weight=0)  # 组名标签
        top_toolbar_frame.grid_columnconfigure(3, weight=0)
        top_toolbar_frame.grid_columnconfigure(4, weight=0)
        top_toolbar_frame.grid_columnconfigure(5, weight=1)#  (占据剩余空间)


        # 1. 文件选择按钮 (模拟“文件”菜单的“打开”功能)
        ttk.Button(
            top_toolbar_frame,
            text="Choose Folder...",
            command=self.select_data_file
        ).grid(row=0, column=0, padx=5, sticky='w')

        # 2. 灰色分隔线 (将文件操作与组选择分隔)
        ttk.Separator(top_toolbar_frame, orient='vertical').grid(row=0, column=1, padx=10, sticky="ns")

        # 3. 组名选择标签
        ttk.Label(top_toolbar_frame, text="Dataset Name:").grid(row=0, column=2, sticky='w')

        # 4. 🗄️ 组名下拉框 (Combobox)
        self.group_combobox = ttk.Combobox(
            top_toolbar_frame,
            textvariable=self.selected_group_name,
            state='readonly',
            width=20,
            values=[]
        )
        self.group_combobox.grid(row=0, column=3, padx=(0, 5), sticky='w')
        self.group_combobox.bind('<<ComboboxSelected>>', self.load_selected_group)

        ttk.Button(
            top_toolbar_frame,
            text="Prediction",
            command=self.start_prediction  # 绑定一个待实现的空方法
        ).grid(row=0, column=4, padx=(0, 5), sticky='w')
        ttk.Label(top_toolbar_frame, text="").grid(row=0, column=5, sticky='ew')


        # ----------------------------------------------
        # II. 菜单/内容区 分隔线 - 占据 Row 1
        # ----------------------------------------------
        separator = ttk.Separator(self, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=3, sticky="ew")

        # ----------------------------------------------
        # III. 核心内容区域布局 (从 Row 2 开始)
        # ----------------------------------------------

        main_row = 2
        self.grid_rowconfigure(main_row, weight=1)
        self.grid_columnconfigure(0, weight=3)  # 左侧通道图占较大空间
        self.grid_columnconfigure(1, weight=3)  # 右侧结果区

        # --- 左侧区域：通道图 (Column 0) ---
        left_frame = ttk.Frame(self, padding="10")
        left_frame.grid(row=main_row, column=0, sticky="nsew")

        ttk.Label(left_frame, text="Real-time Channel Display", font=("Helvetica", 14, "bold")).pack(pady=(10, 10))

        sensor_grid_frame = ttk.Frame(left_frame)
        sensor_grid_frame.pack(expand=True, fill='both', pady=(0, 10))

        for i in range(3):
            sensor_grid_frame.grid_rowconfigure(i, weight=1)
            for j in range(3):
                sensor_grid_frame.grid_columnconfigure(j, weight=1)
                channel_index = i * 3 + j + 1
                container = ttk.Frame(sensor_grid_frame, relief="flat", padding=2)
                container.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")

                img_label = ttk.Label(container, image=self.default_image, text="Waveform", compound="center",
                                      relief="solid")
                img_label.pack(expand=True, fill='both')
                self.sensor_labels.append(img_label)
                ttk.Label(container, text=f"Channel {channel_index}", font=("Arial", 9)).pack()

        # --- 右侧区域：状态 + 预测 + 驾驶图 (Column 1) ---
        right_main_frame = ttk.Frame(self, padding="10")
        right_main_frame.grid(row=main_row, column=1, sticky="nsew")
        right_main_frame.grid_propagate(False)

        # 1. 部件状态部分 (上)
        ttk.Label(right_main_frame, text="Component Status", font=("Helvetica", 14, "bold")).pack(pady=(10, 10))

        status_frame = ttk.Frame(right_main_frame, relief="groove", padding="15")
        status_frame.pack(pady=5, fill='x')

        components = ["Seat cushion", "Steering wheel", "Seat belt", "Pedal"]
        for i, comp in enumerate(components):
            ttk.Label(status_frame, text=f"{comp}:", font=("Arial", 12, "bold")).grid(row=i, column=0, sticky='w',
                                                                                      padx=10, pady=5)

            label_var = tk.StringVar(value="N/A")
            # 【关键】设置 width=15，无论文字多长，Label 宽度不变
            ttk.Label(status_frame, textvariable=label_var, font=("Arial", 12, "bold"),
                      foreground="blue", width=15).grid(row=i, column=1, sticky='w', padx=10)
            self.state_labels[comp] = label_var

        # 2. 预测警示框 (固定宽度)
        # 【关键】设置 width=20，防止 "Warning!!" 撑大容器
        self.result_box = ttk.Label(
            right_main_frame, text="Waiting...", font=("Helvetica", 22, "bold"),
            foreground="white", background="#cccccc", padding="20 60",
            anchor="center", relief="raised", width=20
        )
        self.result_box.pack(pady=(5, 30), fill='x')
        # 3. 驾驶员状态图 (下)
        tk.Label(right_main_frame, text="Driver Status View", font=("Helvetica", 12, "bold")).pack(pady=(10, 2))

        # 创建一个固定大小的容器存放图片
        self.img_container = ttk.Frame(right_main_frame, width=450, height=350)
        self.img_container.pack(pady=(0, 10))
        self.img_container.pack_propagate(False)  # 锁定容器大小

        self.driver_image_label = ttk.Label(
            self.img_container,
            image=self.default_image,
            compound="center",
            anchor="center"
        )
        self.driver_image_label.pack(expand=True, fill='both')

    # ----------------------------------------------
    # 文件选择与分组逻辑分组
    # ----------------------------------------------
    def select_data_file(self):
        folderpath = filedialog.askdirectory(title="选择数据根文件夹")

        if folderpath:
            self.data_groups = self._find_and_group_data_files(folderpath)

            group_names = sorted(self.data_groups.keys())
            self.group_combobox['values'] = group_names

            if group_names:
                default_group = group_names[0]
                self.selected_group_name.set(default_group)
                self.load_selected_group(None)
            else:
                self.selected_group_name.set("-- 未找到数据组 --")
                self.group_combobox['values'] = []
                self.current_filepaths = []

            print(f"找到的数据组：{self.data_groups.keys()}")

    def _find_and_group_data_files(self, root_dir):
        groups = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    base_name, ext = os.path.splitext(filename)
                    group_key = base_name.split('_')[0].split('-')[0]
                    if not group_key:
                        group_key = base_name
                    filepath = os.path.join(dirpath, filename)
                    if group_key not in groups:
                        groups[group_key] = []
                    groups[group_key].append(filepath)
        return groups

    def load_selected_group(self, event):
        group_name = self.selected_group_name.get()
        if group_name in self.data_groups:
            self.current_filepaths = self.data_groups[group_name]
            print(f"已选择数据组 '{group_name}'，包含 {len(self.current_filepaths)} 个文件。")
            # 【待定】下一步：在此处调用处理函数
            # self.update_results(self.current_filepaths)
        else:
            self.current_filepaths = []
            print(f"警告：未找到数据组 '{group_name}' 对应的文件。")

    def start_prediction(self):
        """点击 Prediction 按钮后的逻辑"""
        if not self.current_filepaths:
            print("错误：请先选择数据集")
            return

        # 1. 自动对当前组的文件进行部件分类
        # 这里假设文件名中包含部件关键字（如 'Pedal_1.csv', 'wheel_2.csv'）
        file_mapping = {
            "Pedal": None, "Steering_wheel": None, "Seat_belt": None, "Seat_cushion": []
        }

        for path in self.current_filepaths:
            name = path.lower()
            if "pedal" in name:
                file_mapping["Pedal"] = path
            elif "wheel" in name:
                file_mapping["Steering_wheel"] = path
            elif "belt" in name:
                file_mapping["Seat_belt"] = path
            elif "cushion" in name or "seat" in name:
                file_mapping["Seat_cushion"].append(path)

        # 2. 逐个预测
        results = {"Pedal": "N/A", "Steering_wheel": "N/A", "Seat_belt": "N/A", "Seat_cushion": "N/A"}

        # --- 预测 Pedal, Wheel, Belt (RF模型) ---
        for part in ["Pedal", "Steering_wheel", "Seat_belt"]:
            if file_mapping[part]:
                feat_df = get_features(file_mapping[part])
                if feat_df is not None:
                    results[part] = self.models[part].predict(feat_df)[0]

        # --- 预测 Seat cushion (LSTM模型) ---
        # 注意：LSTM 期望序列输入。如果只有单个文件，我们模拟一个序列
        if file_mapping["Seat_cushion"]:
            # 使用其中一个文件提取特征
            feat_df = get_features(file_mapping["Seat_cushion"][0])
            if feat_df is not None:
                scaler = self.scalers["Seat_cushion"]
                le = self.label_encoders["Seat_cushion"]
                X_scaled = scaler.transform(feat_df)
                # 构造符合 LSTM 输入的 3D 形状 (batch, time_steps, features)
                # 假设 TIME_STEPS = 10，我们把当前特征重复10次作为输入
                X_seq = np.repeat(X_scaled[np.newaxis, :, :], 10, axis=1)
                pred_prob = self.models["Seat_cushion"].predict(X_seq, verbose=0)
                results["Seat_cushion"] = le.inverse_transform([np.argmax(pred_prob)])[0]

        self.update_ui_with_logic(results)

    def update_driver_image(self, status_text):
        """
        根据预测状态更新右下角的驾驶员图片
        :param status_text: 预测结果字符串 (如 "Normal", "Warning！！", "Fatigue")
        """
        folder_name = "Driver's image"
        # 1. 转换逻辑：去除感叹号、转为小写并匹配文件名
        # 例如: "Warning！！" -> "warning.png"
        status_key = status_text.lower().replace("！！", "").strip()
        img_filename = f"{status_key}.png"
        img_path = os.path.join(folder_name, img_filename)

        # 检查文件夹和文件是否存在
        if not os.path.exists(folder_name):
            print(f"错误：找不到文件夹 '{folder_name}'")
            return
        if not os.path.exists(img_path):
            print(f"警告：在 {folder_name} 中找不到文件 '{img_filename}'")
            return

        try:
            # 2. 使用 PIL 打开并缩放图片
            pil_img = Image.open(img_path)

            # 获取 Label 容器的当前尺寸
            # 如果是刚启动尚未显示，给定一个默认参考尺寸 (如 400x300)
            target_w = self.driver_image_label.winfo_width()
            target_h = self.driver_image_label.winfo_height()
            if target_w < 10: target_w, target_h = 450, 350

            # 保持比例缩放 (Image.Resampling.LANCZOS 保证图片清晰)
            pil_img.thumbnail((target_w, target_h), Image.LANCZOS)

            # 3. 转换为 Tkinter 可用的对象
            self.tk_driver_img = ImageTk.PhotoImage(pil_img)
            self.driver_image_label.config(image=self.tk_driver_img, text="")  # 清除占位文字

        except Exception as e:
            print(f"图片加载失败: {e}")

    def update_all_sensor_channels(self, results):
        """
        汇总加载 9 个通道的图片，并确保所有图片大小严格一致。
        """
        root_folder = "channel_image"

        component_map = {
            "Seat_cushion": "Seat cushion",
            "Steering_wheel": "Steering wheel",
            "Seat_belt": "Seat belt",
            "Pedal": "Pedal"
        }

        # --- 1. 设置统一的固定尺寸 ---
        # 根据你的 UI 布局，建议设置为一个固定值（例如 200x140），
        # 这样无论原始图片多大，在 9 宫格里看起来都完全一样。
        fixed_width = 220
        fixed_height = 150

        self.channel_tk_images = [None] * 9

        # 重置 UI
        for i in range(9):
            # 同时也给 Label 设置固定宽高，防止图片加载前后的抖动
            self.sensor_labels[i].config(
                image=self.default_image,
                text=f"Waiting\nCh{i + 1}",
                width=30  # 这里的 width 是字符单位，如果是 ttk.Label 建议主要靠图片撑开
            )

        # 遍历结果
        for res_key, state_text in results.items():
            comp_folder_name = component_map.get(res_key)
            if not comp_folder_name: continue

            # 建议这里对状态文件夹名进行处理，确保匹配（如首字母大写）
            state_folder_name = str(state_text).strip()
            # 自动处理文件夹命名：如果预测是 normal，尝试匹配 "Normal" 或 "normal"
            target_dir = os.path.join(root_folder, comp_folder_name, state_folder_name)

            # 容错：如果找不到，尝试首字母大写
            if not os.path.exists(target_dir):
                target_dir = os.path.join(root_folder, comp_folder_name, state_folder_name.capitalize())

            if not os.path.exists(target_dir):
                continue

            # 扫描目录
            for filename in os.listdir(target_dir):
                match = re.match(r'Channel(\d+)\.png', filename, re.IGNORECASE)
                if match:
                    try:
                        channel_num = int(match.group(1))
                        grid_index = channel_num - 1

                        if 0 <= grid_index < 9:
                            img_path = os.path.join(target_dir, filename)

                            # --- 图片处理 ---
                            pil_img = Image.open(img_path)

                            # 使用 resize 而不是 thumbnail，强制所有图片变为一模一样的大小
                            resample_mode = getattr(Image, 'Resampling', Image).LANCZOS
                            pil_img = pil_img.resize((fixed_width, fixed_height), resample_mode)

                            tk_img = ImageTk.PhotoImage(pil_img)

                            self.channel_tk_images[grid_index] = tk_img
                            # 更新 UI：清除文字，显示统一大小的图片
                            self.sensor_labels[grid_index].config(image=tk_img,
                                                                  text="",
                                                                  anchor="center")

                    except Exception as e:
                        print(f"通道 {filename} 处理失败: {e}")

    def update_ui_with_logic(self, results):
        """更新文字并触发图片切换"""
        # ... 原有的状态提取逻辑 (cushion, wheel, belt, pedal) ...
        cushion = str(results["Seat_cushion"]).lower()
        wheel = str(results["Steering_wheel"]).lower()
        belt = str(results["Seat_belt"]).lower()
        pedal = str(results["Pedal"]).lower()

        # 更新左边具体的文字标签
        self.state_labels["Seat cushion"].set(results["Seat_cushion"])
        self.state_labels["Steering wheel"].set(results["Steering_wheel"])
        self.state_labels["Seat belt"].set(results["Seat_belt"])
        self.state_labels["Pedal"].set(results["Pedal"])

        # 逻辑判断
        final_text = "Attention"
        final_color = "#cccccc"

        if all(s == "normal" for s in [cushion, wheel, belt, pedal]):
            final_text = "Normal"
            final_color = "#28a745"
        elif pedal == "step" and "grip" in wheel and belt == "crush":
            final_text = "Warning！！"
            final_color = "#dc3545"
        elif cushion == "back" and "leave" in wheel and pedal == "normal" and belt == "normal":
            final_text = "Fatigue"
            final_color = "#ffc107"

        # 更新中间的警示框颜色和文字
        self.result_box.config(text=final_text, background=final_color)

        # 【新增】调用图片更新逻辑
        self.update_driver_image(final_text)
        self.update_all_sensor_channels(results)



if __name__ == "__main__":
    app = Application()
    app.mainloop()
