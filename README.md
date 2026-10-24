# CatBreedID – Nhận diện 12 Giống Mèo & Giải thích Vùng chú ý (Grad-CAM)

> **Môn học:** Thị giác máy tính và ứng dụng (422001503102)  
> **Lớp:** DHKHMT19BTT – **Nhóm thực hiện:** Nhóm 09  
> **Ứng dụng:** Mô phỏng giao diện người dùng quy trình phân loại giống mèo, kiểm chứng từ chối ảnh ngoại lai (Out-of-Distribution - OOD) và giải thích quyết định mạng nơ-ron bằng Grad-CAM.

---

## 📌 Giới thiệu dự án

Hệ thống được xây dựng nhằm giải quyết bài toán nhận diện giống vật nuôi trong thực tế:
1. **Phân loại giống:** Nhận diện 12 giống mèo phổ biến thuộc bộ dữ liệu Oxford-IIIT Pet Dataset (*Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx*).
2. **Nhận dạng tập mở (Open-Set Recognition / OOD Detection):** Tích hợp 5 phương pháp hậu kiểm (MSP, MaxLogit, Energy Score, Mahalanobis Feature Space, KNN Distance) để từ chối các đối tượng ngoại lai (loài khác, chó, đồ vật, ảnh nhiễu) thay vì phỏng đoán sai.
3. **Giải thích mô hình (Explainable AI - XAI):** Sử dụng kỹ thuật Grad-CAM trực quan hóa bản đồ nhiệt (Heatmap) trên ảnh gốc, giúp người dùng biết mô hình đã dựa vào đặc trưng nào (tai, mắt, sống mũi, lông) để đưa ra quyết định.

---

## 🚀 Cách chạy ứng dụng

Giao diện được xây dựng hoàn toàn bằng **HTML5 / CSS3 / Vanilla JavaScript chuẩn**, không cần cài đặt môi trường build phức tạp hay npm dependencies.

### Cách 1: Chạy trực tiếp
- Mở thư mục dự án trên máy tính.
- Nhấp đúp chuột vào file `index.html` để mở ngay trên trình duyệt (Chrome, Edge, Firefox, Safari).

### Cách 2: Sử dụng Live Server trong VS Code (Khuyên dùng)
1. Mở thư mục dự án bằng **Visual Studio Code**.
2. Cài đặt tiện ích mở rộng **Live Server** (nếu chưa có).
3. Nhấp chuột phải vào `index.html` $\rightarrow$ chọn **"Open with Live Server"** (mặc định tại `http://127.0.0.1:5500`).

### Cách 3: Chạy bằng máy chủ Python nội bộ
Mở Terminal/PowerShell tại thư mục `Interface`:
```bash
python -m http.server 8080
```
Sau đó truy cập trình duyệt tại địa chỉ: [http://localhost:8080](http://localhost:8080)

---

## 🎯 Hướng dẫn trải nghiệm các tính năng

### 1. Thử nhanh với 2 ảnh mẫu kiểm chứng đồ án
Ngay dưới vùng tải ảnh ở trang chủ, bạn có thể click chọn nhanh 2 ảnh mẫu đại diện cho 2 kịch bản thực tế:
- **`Abyssinian_1.jpg (Mèo • Khá chắc chắn)`**:
  - Mô phỏng trường hợp ảnh hợp lệ (**In-Distribution**).
  - Nhấn **"Phân tích ảnh"** $\rightarrow$ Kết quả nhận diện đúng giống *Abyssinian* với độ tự tin **94%**.
  - Phần **Grad-CAM** hiển thị bản đồ nhiệt tập trung chuẩn xác vào khuôn mặt và đôi tai mèo.
  - Bảng **Chế độ nâng cao** thể hiện **4/5 phương pháp OOD chấp nhận**.
- **`american_bulldog_2.jpg (Chó • Chưa xác định)`**:
  - Mô phỏng trường hợp ảnh ngoại lai (**Near-OOD** - loài chó gần với mèo nhưng không nằm trong tập huấn luyện).
  - Nhấn **"Phân tích ảnh"** $\rightarrow$ Hệ thống kích hoạt cơ chế từ chối, hiển thị nhãn **Chưa xác định** (28% độ tin cậy).
  - Tự động ẩn Grad-CAM và hiện bảng gợi ý chụp ảnh chuẩn.
  - Bảng **Chế độ nâng cao** thể hiện **5/5 phương pháp OOD từ chối**.

### 2. Tải ảnh từ thiết bị
- Kéo thả file ảnh (JPG, PNG) trực tiếp vào khung nét đứt hoặc nhấn nút **"Chọn ảnh từ máy"**.
- Xem trước ảnh đã chọn và nhấn **"Phân tích ảnh"**.
- Nhấn nút **"Chọn ảnh khác"** bất cứ lúc nào để quay lại màn hình chính.

### 3. Tương tác với vùng chú ý Grad-CAM
- **Bật/Tắt vùng chú ý:** Nhấp vào ô checkbox *"Hiện vùng máy chú ý"*.
- **Tùy chỉnh độ đậm nhiệt:** Kéo thanh trượt *"Độ đậm nhiệt"* (0% – 100%) để quan sát rõ tương quan giữa bản đồ nhiệt và chi tiết ảnh gốc bên dưới.

### 4. Đánh giá kỹ thuật chuyên sâu & Chế độ nâng cao
Cuộn xuống phần dưới của trang kết quả để xem:
- Bảng **Thông tin kỹ thuật chuyên sâu:** Ngưỡng quyết định ($\tau$), kiến trúc mạng nơ-ron nền tảng, thời gian trễ suy luận.
- Bảng **Chế độ nâng cao:** So sánh chi tiết điểm số đo được, ngưỡng cắt và kết luận của 5 thuật toán phát hiện ngoại lai.

---

## 📂 Cấu trúc thư mục

```text
Interface/
├── index.html              # Cấu trúc giao diện HTML5 chuẩn semantic
├── README.md               # Hướng dẫn sử dụng & thông tin dự án
├── .gitignore              # Cấu hình bỏ qua các file tạm, môi trường ảo
├── css/
│   └── style.css           # Toàn bộ CSS (Hệ thống thiết kế, bố cục 1040px, responsive, WCAG 2.2)
├── js/
│   └── app.js              # Logic điều khiển, luồng sự kiện phân tích, animation, tương tác Grad-CAM
└── img/
    ├── Abyssinian_1.jpg     # Ảnh mẫu mèo In-Distribution dùng để kiểm chứng
    └── american_bulldog_2.jpg # Ảnh mẫu chó Near-OOD dùng để kiểm chứng từ chối
```

---

## 👥 Nhóm tác giả
- **Đồ án:** Thị giác máy tính và ứng dụng
- **Lớp:** DHKHMT19BTT
- **Nhóm:** 09
