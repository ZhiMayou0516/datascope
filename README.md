# DataScope AutoML Report

DataScope AutoML Report 是一个面向科研与工程数据的轻量级自动分析、建模与报告生成工具。项目以 Streamlit 为交互界面，围绕表格数据、影像数组数据和基础深度学习原型，提供从数据读取、质量检查、探索性分析、机器学习建模、可解释性分析到报告导出的完整流程。

项目采用本地优先的设计方式，不依赖外部 API，适合在个人电脑或实验室环境中快速完成数据初筛、建模验证和结果整理。

## 主要功能

### 1. 多格式数据读取

- 支持 `CSV`、`TSV`、`TXT`
- 支持 Excel 文件：`xlsx`、`xls`
- 支持 Excel 多 sheet 选择
- 支持大文件上传配置，默认上限提高到 2048 MB
- 支持通过本机路径直接读取大体积数据，减少浏览器上传压力
- 内置分类、回归和影像 NPZ 示例数据，便于快速测试

### 2. 自动 EDA 分析

- 数据规模、缺失值、重复行、内存占用统计
- 数值变量分布分析
- 类别变量频数统计
- 相关性热图
- 目标变量分布可视化
- 自动识别数值列、类别列、日期列、布尔列和高基数类别列

### 3. 数据质量检查

- IQR 异常值检测
- 高基数类别特征识别
- 长尾类别合并策略
- 目标列质量检查
- 潜在数据泄漏风险提示
- 对异常字段给出可解释的提示，便于后续人工核查

### 4. 机器学习建模

支持分类与回归任务，当前包含：

- Logistic Regression
- Random Forest
- Gradient Boosting
- Ridge / Linear Regression
- XGBoost 接口
- LightGBM 接口

建模流程包括：

- 自动划分训练集与测试集
- 数值特征缺失值填补与标准化
- 类别特征缺失值填补、长尾类别合并与 One-Hot 编码
- 分类任务输出 Accuracy、Precision、Recall、F1、AUC 等指标
- 回归任务输出 MAE、RMSE、R² 等指标
- 特征重要性提取
- 预测结果表格导出

### 5. 交叉验证与超参数搜索

- 支持 K 折交叉验证
- 支持 Grid Search
- 支持 Random Search
- 自动根据任务类型选择合适的评估指标
- 输出交叉验证结果和最佳参数搜索结果
- 避免只依赖单次训练/测试划分带来的偶然性

### 6. SHAP 可解释性分析

- 支持对已训练模型进行 SHAP 特征贡献分析
- 输出 SHAP 全局重要性表格
- 提供特征贡献可视化
- 当环境缺少 SHAP 或模型暂不支持时，会给出明确提示

### 7. 影像与数组数据支持

项目额外加入了常见科研影像数据入口：

- 支持读取 `NPZ` 文件
- 自动展示 NPZ 内部数组名称、维度、数据类型、取值范围
- 支持数组切片预览
- 支持将 NPZ 影像数组转换为统计特征表，并接入表格建模流程
- 支持读取 DICOM 元数据
- 支持 DICOM 像素矩阵预览
- 提供轻量级 CNN / RNN / LSTM / ViT 原型训练入口

### 8. 自动报告生成

- 一键生成 Markdown 报告
- 一键生成 HTML 报告
- HTML 报告可嵌入 EDA 图表
- 报告内容包括：
  - 数据概览
  - 缺失值统计
  - 字段类型识别
  - 数据质量检查
  - 模型配置
  - 模型指标
  - 交叉验证结果
  - 特征重要性
  - 结论摘要

### 9. 配置保存与复现实验

- 保存当前数据来源、目标列、任务类型和模型参数
- 导出 `datascope_config.json`
- 支持重新加载配置文件
- 方便复现实验流程和对比不同建模方案

## 项目结构

```text
datascope-automl-report-v3/
├── app.py                         # Streamlit 主程序
├── requirements.txt               # 完整依赖
├── requirements-lite.txt          # 轻量依赖
├── run_app.bat                    # Windows 一键启动脚本
├── run_app.sh                     # Linux / macOS 启动脚本
├── VERSION.txt                    # 版本说明
├── .streamlit/
│   └── config.toml                # Streamlit 上传大小等配置
├── sample_data/
│   ├── classification_demo.csv    # 分类示例数据
│   ├── regression_demo.csv        # 回归示例数据
│   └── imaging_npz_demo.npz       # 影像数组示例数据
├── src/
│   ├── config_manager.py          # 配置保存与读取
│   ├── data_loader.py             # 表格数据读取
│   ├── deep_models.py             # CNN/RNN/LSTM/ViT 原型模型
│   ├── eda.py                     # EDA 与数据质量分析
│   ├── explainability.py          # SHAP 可解释性分析
│   ├── imaging.py                 # NPZ / DICOM 数据处理
│   ├── modeling.py                # 机器学习建模流程
│   ├── report.py                  # Markdown / HTML 报告生成
│   ├── utils.py                   # 通用工具函数
│   └── visualization.py           # 可视化函数
└── outputs/
    └── .gitkeep                   # 报告与配置输出目录
```

## 安装方式

建议使用 Conda 或虚拟环境运行。

```bash
conda create -n datascope python=3.10
conda activate datascope
```

安装基础依赖：

```bash
pip install -r requirements-lite.txt
```

安装完整依赖：

```bash
pip install -r requirements.txt
```

完整依赖包含 XGBoost、LightGBM、SHAP、PyTorch 和 pydicom。如果只是运行表格 EDA 与基础机器学习流程，可以先使用轻量依赖。

## 运行方式

### Windows

```bash
run_app.bat
```

或者手动运行：

```bash
streamlit run app.py
```

### Linux / macOS

```bash
bash run_app.sh
```

或者：

```bash
streamlit run app.py
```

运行后，浏览器会打开本地页面：

```text
http://localhost:8501
```


## 示例场景

### 表格分类任务

适用于包含样本特征和分类标签的数据，例如：

- 疾病阳性/阴性预测
- 实验组别分类
- 风险等级分类
- 材料或样品类别识别

### 表格回归任务

适用于预测连续数值结果，例如：

- 指标浓度预测
- 实验测量值预测
- 评分或性能指标预测
- 时间、强度、响应值等连续变量预测

### 影像数组任务

适用于初步处理 `.npz` 或 DICOM 数据，例如：

- 查看数组维度与数据范围
- 快速预览影像切片
- 将影像数组提取为统计特征表
- 使用轻量 CNN / RNN / LSTM / ViT 做原型验证

## 大文件处理说明

项目已经在 `.streamlit/config.toml` 中提高上传限制：

```toml
[server]
maxUploadSize = 2048
maxMessageSize = 2048
```

对于非常大的科研数据，更推荐使用 `Upload Data` 页面中的本机路径读取功能，例如：

```text
D:/data/my_dataset.csv
```

这样可以避免浏览器上传大文件时占用过多内存。

## 输出文件

运行过程中产生的报告与配置文件默认保存在：

```text
outputs/
```

常见输出包括：

```text
datascope_report.md
datascope_report.html
datascope_config.json
```

## 技术栈

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly
- Jinja2
- XGBoost
- LightGBM
- SHAP
- PyTorch
- pydicom

其中 XGBoost、LightGBM、SHAP、PyTorch 和 pydicom 属于可选增强依赖。缺少这些库时，项目仍然可以运行基础表格分析和建模功能。

## 设计思路

DataScope 的核心目标不是替代完整的数据科学平台，而是提供一个轻量、可读、可改、可本地运行的数据分析工作台。它更关注以下几点：

- 快速读取不同来源的数据
- 自动完成基础 EDA 与数据质量检查
- 用统一流程完成 baseline 建模
- 给出可解释的模型结果
- 将分析过程整理成可下载报告
- 为影像数组和深度学习原型保留扩展入口

## 后续可扩展方向

- 增加更多模型，例如 CatBoost、TabNet 等
- 增加更完整的时间序列建模模块
- 增加医学影像分割或分类专用数据管线
- 增加模型保存与加载功能
- 增加批量实验管理
- 增加报告模板自定义功能
- 增加 Docker 部署方式
- 增加更细粒度的数据泄漏检测规则

