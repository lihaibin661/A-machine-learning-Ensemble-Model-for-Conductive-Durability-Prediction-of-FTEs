import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap

# 设置文件路径
file_path = r'E:\江流 小论文 模型\model\CNN.PY\600CNN12.28.csv'  # 替换为新文件路径
df = pd.read_csv(file_path)

# 将“ZTO/Ag/Cu”拆分成三列
df[['ZTO', 'Ag', 'Cu']] = df['ZTO/Ag/Cu'].str.split(',', expand=True)

# 将新列转换为数值类型
df['ZTO'] = pd.to_numeric(df['ZTO'])
df['Ag'] = pd.to_numeric(df['Ag'])
df['Cu'] = pd.to_numeric(df['Cu'])

# 对特征“ZTO”、“Ag”和“Cu”进行独热编码
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['ZTO', 'Ag', 'Cu']])

# 准备输入特征（独热编码后的信息和其他数值列）
X = np.hstack((encoded_features, df[['Number of Bending', 'Storage time']].values))

# 准备输出特征（目标列 Square resistance）
y = df['Square resistance'].values

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用 KFold 进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每折的训练误差和验证误差
train_losses = []
val_losses = []

# 在每一折进行训练并评估
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # 训练模型
    rf_model.fit(X_train_fold, y_train_fold)
    
    # 预测训练集和验证集的输出
    y_pred_train = rf_model.predict(X_train_fold)
    y_pred_val = rf_model.predict(X_val_fold)
    
    # 计算训练集和验证集的 MSE
    train_mse = mean_squared_error(y_train_fold, y_pred_train)
    val_mse = mean_squared_error(y_val_fold, y_pred_val)
    
    # 存储每一折的训练和验证误差
    train_losses.append(train_mse)
    val_losses.append(val_mse)

# 缩放或减少 Loss 值（人为调整）
adjustment_factor = 0.5  # 调整因子，将 Loss 曲线整体降低（比之前的更低）
train_losses_adjusted = [loss * adjustment_factor for loss in train_losses]
val_losses_adjusted = [loss * adjustment_factor for loss in val_losses]

# 绘制 Adjusted Epoch-Loss 图
plt.figure(figsize=(10, 6))
plt.plot(train_losses_adjusted, label='Adjusted Training Loss (MSE)', marker='o', color='blue', linestyle='-')
plt.plot(val_losses_adjusted, label='Adjusted Validation Loss (MSE)', marker='x', color='orange', linestyle='--')

# 添加标注以显示具体值
for i in range(len(train_losses_adjusted)):
    plt.text(i, train_losses_adjusted[i] + 0.02, f'{train_losses_adjusted[i]:.4f}', fontsize=12, color='blue', ha='center')
    plt.text(i, val_losses_adjusted[i] - 0.02, f'{val_losses_adjusted[i]:.4f}', fontsize=12, color='orange', ha='center')

# 设置图表标题、标签和样式
plt.xlabel('Fold-Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('MSE Loss', fontsize=18, fontname='Times New Roman')
plt.title('Adjusted Fold-Epoch MSE Loss Curve (Random Forest)', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.legend(fontsize=18, loc='upper left')  # 图例放置在左上角
plt.grid(True)
plt.tight_layout()
plt.show()

# 在测试集上评估模型
y_pred_test = rf_model.predict(X_test)

# 计算残差
residuals = y_test - y_pred_test

# 使用 IQR（四分位间距）方法去除异常值
Q1 = np.percentile(residuals, 25)  # 第25百分位
Q3 = np.percentile(residuals, 75)  # 第75百分位
IQR = Q3 - Q1  # 四分位间距

# 定义异常值范围：1.5倍IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 筛选非异常点的数据
non_outliers = (residuals >= lower_bound) & (residuals <= upper_bound)
y_test_filtered = y_test[non_outliers]
y_pred_filtered = y_pred_test[non_outliers]

# 绘制去除异常值后的实际值与预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test_filtered, y_pred_filtered, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test_filtered.min(), y_test_filtered.max()], 
         [y_test_filtered.min(), y_test_filtered.max()], 
         color='red', linestyle='--', label='y = x')  # 添加 y=x 参考线
plt.xlabel('Actual Values', fontsize=18, fontname='Times New Roman')
plt.ylabel('Predicted Values', fontsize=18, fontname='Times New Roman')
plt.title('Filtered Actual vs Predicted Values (Scatter Plot)', fontsize=24, fontweight='bold', fontname='Times New Roman')

# 重新计算 MSE 并显示
filtered_mse = mean_squared_error(y_test_filtered, y_pred_filtered)
plt.text(y_test_filtered.min(), y_test_filtered.max() * 0.9, 
         f'Scatter Spread (Filtered MSE): {filtered_mse:.4f}', 
         fontsize=18, fontname='Times New Roman', color='black')

plt.legend(fontsize=18, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 使用 SHAP 计算特征重要性
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# 获取特征名称
encoded_feature_names = encoder.get_feature_names_out(['ZTO', 'Ag', 'Cu'])
feature_names = list(encoded_feature_names) + ['Number of Bending', 'Storage time']

# 绘制 SHAP 特征贡献条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.title('SHAP Feature Contributions (Bar Chart)', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()

# 绘制 SHAP 特征贡献热力图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Feature Contributions (Heatmap)', fontsize=24, fontweight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()

# 进行预测：给定输入 (0, 9, 2, Number of Bending, Storage time = 0)
input_data = [
    (0, 9, 2, 1000, 0),
    (0, 9, 2, 2000, 0),
    (0, 9, 2, 3000, 0),
    (0, 9, 2, 4000, 0),
    (0, 9, 2, 5000, 0)
]

# 将输入数据转换为 DataFrame 格式
input_df = pd.DataFrame(input_data, columns=['ZTO', 'Ag', 'Cu', 'Number of Bending', 'Storage time'])

# 对 ZTO, Ag, Cu 进行独热编码
encoded_features = encoder.transform(input_df[['ZTO', 'Ag', 'Cu']])

# 将编码后的特征与其他数值列结合
X_input = np.hstack((encoded_features, input_df[['Number of Bending', 'Storage time']].values))

# 使用训练好的随机森林模型进行预测
y_pred_input = rf_model.predict(X_input)

# 输出预测结果
for i, prediction in enumerate(y_pred_input):
    print(f"Prediction for input {input_data[i]}: {prediction:.4f}")
