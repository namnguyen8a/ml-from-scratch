import numpy as np
from MultipleLinearRegression import MultipleLinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data/advertising.csv")

X = df[["TV", "Radio", "Newspaper"]].values
y = df["Sales"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled.shape)
print(y.shape)

reg = MultipleLinearRegression(lr=0.01, n_iters=1000) # Try a slightly larger learning rate after scaling
w, b, losses = reg.fit(X_scaled, y)

print("Trọng số sau huấn luyện:", w)
print("Bias sau huấn luyện:", b)
print("Mất mát cuối cùng:", losses[-1])

# 1. Biểu đồ Mất mát theo số lượng Iteration
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses)
plt.xlabel("Số lượng Iteration")
plt.ylabel("Mất mát (Loss)")
plt.title("Biểu đồ Mất mát trong quá trình Huấn luyện")
plt.grid(True)
plt.show()

# 2. Biểu đồ Đường thẳng hồi quy (chỉ với một đặc trưng 'TV')
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='blue', label='Dữ liệu Thực tế (TV vs Sales)')

# Lấy min và max của 'TV' chưa scale
x_min_tv = np.min(X[:, 0])
x_max_tv = np.max(X[:, 0])
X_plot_tv = np.array([[x_min_tv], [x_max_tv]])

# Chuẩn hóa các điểm vẽ 'TV' bằng scaler đã fit trên toàn bộ X
X_plot_scaled_tv = scaler.transform(np.concatenate([X_plot_tv, np.zeros((2, 2))], axis=1))[:, 0:1]

# Dự đoán (lưu ý rằng bias đã được học từ dữ liệu đã chuẩn hóa)
y_predict_scaled = np.dot(X_plot_scaled_tv, w[0]) + b

# Chuyển đổi dự đoán về thang đo ban đầu của y (nếu y đã được scale)
y_predict = y_predict_scaled # Nếu y không được scale riêng

plt.plot(X_plot_tv, y_predict, color='red', label='Đường thẳng Hồi quy (ước tính từ mô hình đa biến)')
plt.xlabel("Chi phí TV")
plt.ylabel("Doanh số")
plt.title("Đường thẳng Hồi quy trên Dữ liệu (TV vs Sales)")
plt.legend()
plt.grid(True)
plt.show()

# Nếu bạn muốn thử với 2 đặc trưng (vẫn đơn giản hơn 3D):
# X_2_features = df[["TV", "Radio"]].values
# scaler_X_2 = StandardScaler()
# X_scaled_2 = scaler_X_2.fit_transform(X_2_features)
# reg_2 = LinearRegression(lr=0.01, n_iters=1000)
# w_2, b_2, losses_2 = reg_2.fit(X_scaled_2, y_scaled)
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_2_features[:, 0], X_2_features[:, 1], y, c='blue', marker='o', label='Dữ liệu Thực tế')
#
# # Tạo lưới điểm để vẽ mặt phẳng (ước tính) - phức tạp hơn và có thể bỏ qua nếu chỉ cần điểm dự đoán
#
# predicted_y_2 = scaler_y.inverse_transform(reg_2.predict(X_scaled_2))
# ax.scatter(X_2_features[:, 0], X_2_features[:, 1], predicted_y_2.flatten(), c='red', marker='^', label='Dự đoán')
#
# ax.set_xlabel('Chi phí TV')
# ax.set_ylabel('Chi phí Radio')
# ax.set_zlabel('Doanh số')
# ax.legend()
# plt.show()