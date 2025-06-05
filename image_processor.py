import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns # Để vẽ biểu đồ đẹp hơn

# --- Phần 1: Thiết lập Đường dẫn và Tải Dữ liệu (Mô phỏng Big Data - Nâng cấp) ---
base_data_dir = "." # Ký tự '.' đại diện cho thư mục hiện tại

categories = ['animals', 'objects', 'flowers'] # Các danh mục bạn muốn phân loại

image_data = []

for category in categories:
    path = os.path.join(base_data_dir, category)
    if os.path.isdir(path):
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(path, img_name)
                image_data.append({
                    'image_path': img_path,
                    'category': category,
                    'file_name': img_name,
                    'uploaded_by_user_id': np.random.randint(1, 101),
                    'upload_timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
                })
    else:
        print(f"Warning: Directory not found or not a directory: {path}")

df = pd.DataFrame(image_data)

if df.empty:
    print("Không tìm thấy hình ảnh nào để xử lý. Vui lòng kiểm tra lại thư mục và các file ảnh.")
    exit()

print("DataFrame ban đầu (với dữ liệu mô phỏng user và thời gian):")
print(df.head())
print(f"\nTổng số hình ảnh được tìm thấy: {len(df)}")

# --- Phần 2: Tích hợp Mô hình AI (Thị giác máy tính) ---

print("\nĐang tải mô hình MobileNetV2... (Có thể mất thời gian nếu tải lần đầu)")
model = MobileNetV2(weights='imagenet')
print("Đã tải xong mô hình MobileNetV2.")

df['ai_predicted_category'] = None
df['ai_prediction_confidence'] = None
df['ai_top_3_predictions'] = None

print("\nBắt đầu phân loại hình ảnh bằng AI (Thị giác máy tính)...")
start_time_ai = time.time()

for index, row in df.iterrows():
    img_path = row['image_path']
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded_dims)

        predictions = model.predict(img_preprocessed, verbose=0) # verbose=0 để không in ra tiến độ mỗi ảnh
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        df.at[index, 'ai_predicted_category'] = decoded_predictions[0][1]
        df.at[index, 'ai_prediction_confidence'] = decoded_predictions[0][2]
        df.at[index, 'ai_top_3_predictions'] = [pred[1] for pred in decoded_predictions]

        # In ra thông báo cho mỗi ảnh để theo dõi tiến độ
        print(f"Đã xử lý: {row['file_name']} -> Dự đoán: {decoded_predictions[0][1]} ({decoded_predictions[0][2]:.2f})")

    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")
        df.at[index, 'ai_predicted_category'] = "Error"
        df.at[index, 'ai_prediction_confidence'] = 0
        df.at[index, 'ai_top_3_predictions'] = []

end_time_ai = time.time()
print(f"\nĐã hoàn thành phân loại hình ảnh trong {end_time_ai - start_time_ai:.2f} giây.")

print("\nDataFrame sau khi phân loại bằng AI (5 dòng đầu tiên):")
print(df.head())

# --- Phần 3: Lưu trữ Dữ liệu Lớn (Kết quả AI) ---
output_csv_path = os.path.join(base_data_dir, 'image_metadata_with_ai.csv')
df.to_csv(output_csv_path, index=False)
print(f"\nĐã lưu dữ liệu hình ảnh (có kết quả AI) vào: {output_csv_path}")

# --- Phần 4: Phân tích và Trực quan hóa Dữ liệu (Mô phỏng Big Data Analytics) ---

print("\n--- Bắt đầu Phân tích Dữ liệu Lớn (Kết quả AI) ---")

# 4.1 Thống kê cơ bản theo danh mục được AI dự đoán
print("\nThống kê số lượng hình ảnh theo danh mục AI dự đoán:")
ai_category_counts = df['ai_predicted_category'].value_counts()
print(ai_category_counts)

print("\nThống kê phần trăm hình ảnh theo danh mục AI dự đoán:")
ai_category_percentages = df['ai_predicted_category'].value_counts(normalize=True) * 100
print(ai_category_percentages)

# 4.2 Thống kê theo user (mô phỏng)
print("\nThống kê số lượng ảnh đã upload theo từng User ID:")
user_upload_counts = df['uploaded_by_user_id'].value_counts().sort_index()
print(user_upload_counts.head()) # In ra 5 user đầu tiên để không quá dài
print(f"Tổng số User khác nhau: {df['uploaded_by_user_id'].nunique()}")


# 4.3 Trực quan hóa dữ liệu
print("\nĐang tạo biểu đồ...")

# Cấu hình để hiển thị tiếng Việt trong biểu đồ (nếu cần)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Hoặc 'Arial', 'Times New Roman' nếu bạn có font tiếng Việt
plt.rcParams['axes.unicode_minus'] = False

# Biểu đồ cột: Số lượng hình ảnh theo danh mục AI dự đoán
plt.figure(figsize=(12, 6))
sns.barplot(x=ai_category_counts.index, y=ai_category_counts.values, palette='viridis')
plt.title('Số lượng hình ảnh theo Danh mục AI dự đoán')
plt.xlabel('Danh mục AI dự đoán')
plt.ylabel('Số lượng hình ảnh')
plt.xticks(rotation=45, ha='right') # Xoay tên danh mục nếu dài
plt.tight_layout() # Điều chỉnh layout để không bị che khuất
plt.savefig(os.path.join(base_data_dir, 'ai_category_counts_bar_chart.png'))
print("Đã lưu biểu đồ cột: ai_category_counts_bar_chart.png")
# plt.show() # Uncomment dòng này nếu bạn muốn hiển thị biểu đồ ngay lập tức

# Biểu đồ tròn: Phần trăm hình ảnh theo danh mục AI dự đoán
plt.figure(figsize=(10, 10))
plt.pie(ai_category_percentages, labels=ai_category_percentages.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Phần trăm hình ảnh theo Danh mục AI dự đoán')
plt.axis('equal') # Đảm bảo biểu đồ tròn không bị méo
plt.savefig(os.path.join(base_data_dir, 'ai_category_percentages_pie_chart.png'))
print("Đã lưu biểu đồ tròn: ai_category_percentages_pie_chart.png")
# plt.show() # Uncomment dòng này nếu bạn muốn hiển thị biểu đồ ngay lập tức

# Biểu đồ cột: Số lượng ảnh upload theo User ID (top N users)
# Chúng ta chỉ vẽ top 10 user để biểu đồ không quá dày đặc
top_n_users = user_upload_counts.nlargest(10)
if not top_n_users.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_n_users.index.astype(str), y=top_n_users.values, palette='coolwarm')
    plt.title('Top 10 User upload nhiều ảnh nhất')
    plt.xlabel('User ID')
    plt.ylabel('Số lượng ảnh đã upload')
    plt.tight_layout()
    plt.savefig(os.path.join(base_data_dir, 'top_user_uploads_bar_chart.png'))
    print("Đã lưu biểu đồ cột User Uploads: top_user_uploads_bar_chart.png")
    # plt.show()
else:
    print("Không đủ dữ liệu để vẽ biểu đồ Top User Uploads.")


print("\n--- Hoàn thành Phân tích và Trực quan hóa Dữ liệu ---")