# FRUIT SLASH 🍉

Trò chơi chém trái cây bằng camera và cử chỉ hai bàn tay, chạy trực tiếp trên Windows bằng Python.

## Tính năng
- Camera trực tiếp và nhận diện người bằng MediaPipe Pose.
- Theo dõi tối đa hai bàn tay bằng MediaPipe Hands.
- Dùng đầu ngón trỏ để kích hoạt nút bắt đầu (giữ khoảng 1 giây).
- Đếm ngược 3–2–1 trước khi chơi.
- Chém trái cây bằng chuyển động tay; trái cây tách đôi và tạo hiệu ứng hạt.
- Bom xuất hiện ngẫu nhiên; chém trúng bom sẽ kết thúc lượt.
- Điểm số, ba mạng, cấp độ và độ khó tăng theo thời gian.
- Tạm dừng, tiếp tục, chơi lại và thoát.

## Cài đặt nhanh (Windows 10/11)
1. Cài Python **3.11 64-bit** và VS Code.
2. Clone hoặc tải repository này về máy.
3. Mở thư mục bằng VS Code.
4. Chạy `run.bat`. Script sẽ tạo môi trường ảo, cài thư viện và chạy game.

## Chạy thủ công
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Nếu PowerShell không cho kích hoạt môi trường ảo, dùng trực tiếp:
```powershell
.\.venv\Scripts\python.exe main.py
```

## Điều khiển
- Đứng trước camera sao cho thấy rõ thân trên và hai cánh tay.
- Trỏ ngón trỏ vào nút **BẮT ĐẦU TRÒ CHƠI** ở vùng chơi và giữ yên khoảng 1 giây.
- Sau đếm ngược, vung một hoặc hai tay để đường di chuyển của ngón trỏ cắt ngang trái cây.
- Tránh bom. Bỏ lỡ trái cây sẽ mất mạng.
- `P`: tạm dừng/tiếp tục.
- `SPACE`: bắt đầu hoặc chơi lại từ màn hình menu/kết thúc.
- `ESC`: thoát.

## Khắc phục sự cố camera
- Đóng Zoom, Teams, Camera hoặc ứng dụng khác đang chiếm webcam.
- Vào Windows Settings → Privacy & security → Camera và bật quyền truy cập cho desktop apps.
- Nếu có nhiều camera, sửa `CAMERA_INDEX = 0` trong `main.py` thành `1` hoặc `2`.

## Công nghệ
Python 3.11 · Pygame · OpenCV · MediaPipe Pose · MediaPipe Hands

Camera và nhận diện được xử lý ngay trên máy, không tải hình ảnh lên máy chủ. Phiên bản hiện tại dùng đồ họa trái cây vẽ bằng Pygame (chưa phải ảnh chụp thật).
