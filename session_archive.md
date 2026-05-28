# 会话存档 — 2026-05-28

## 项目概述

为 Ryan Donnelly 水凝胶微针（HFMN）文献数据集搭建了完整的机器学习预测项目。

- **数据来源**：`Ryan_Donnelly_HFMN_AI_dataset_firstpass.xlsx`，sheet 名 `Extracted_Data`
- **数据规模**：25 行 × 45 列
- **GitHub 仓库**：https://github.com/BMZ-Shu/HFMN_AI_Predictor
- **GitHub 账号**：BMZ-Shu

---

## 项目结构

```
C:\1\AI\
├── requirements.txt              # Python 依赖
├── train_model.py                # 模型训练脚本
├── app.py                        # Streamlit 交互式应用
├── README.md                     # 项目说明文档
├── .gitignore                    # Git 忽略规则
├── session_archive.md            # 本文件（会话存档）
├── Ryan_Donnelly_HFMN_AI_dataset_firstpass.xlsx   # 原始数据
├── models/                       # 训练好的模型文件
│   ├── model_swelling_pct_mean.joblib
│   ├── model_gel_fraction_pct_mean.joblib
│   └── model_Fmax_N_per_cm2_mean.joblib
└── outputs/                      # 输出结果和图表
    ├── model_performance.xlsx
    ├── feature_importance_each_target.xlsx
    ├── actual_vs_predicted_*.png（3 张）
    └── feature_importance_*.png（3 张）
```

---

## 技术细节

### 输入特征（15 个候选，实际可用约 7 个）
- **配方**：PVA_wt_pct, PVP_wt_pct, citric_acid_wt_pct, PMVE_MA_or_Gantrez_wt_pct, PEG10000_wt_pct, PEG400_wt_pct, glycerol_wt_pct, water_wt_pct
- **工艺**：crosslink_temp_C, crosslink_time_min
- **几何**：needle_number_total, needle_height_um, needle_base_width_um, interspacing_um, film_thickness_mm

### 预测目标
- swelling_pct_mean（溶胀百分比）
- gel_fraction_pct_mean（凝胶分数）
- Fmax_N_per_cm2_mean（最大力学强度 N/cm²）

### 处理方法
- 缺失值：中位数填充（SimpleImputer, strategy=median）
- 全空列自动丢弃（scikit-learn keep_empty_features=False）
- 字符串清洗（如 `>600` → `600`）
- 最少 5 个有效样本才训练

### 交叉验证
- 留一法（LOO），因为样本数 25 < 30
- 使用 cross_val_predict 收集所有预测值后统一计算 R²/MAE/RMSE

### 模型对比结果

| 目标 | LinearRegression R² | RandomForest R² |
|------|--------------------|------------------|
| swelling_pct_mean | 0.34 | 0.09 |
| gel_fraction_pct_mean | 0.86 | 0.71 |
| Fmax_N_per_cm2_mean | 0.89 | 0.77 |

### 最终模型
- RandomForestRegressor（n_estimators=200, max_depth=5, random_state=42）

---

## 常用命令

```powershell
# 进入项目
cd C:\1\AI

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train_model.py

# 启动 Streamlit 应用
streamlit run app.py

# 推送到 GitHub
git add -A
git commit -m "更新说明"
git push
```

---

## 全局偏好

- 永远使用中文交流（已写入 C:\Users\37670\.claude\CLAUDE.md）
