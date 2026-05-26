# Phân tích luồng hoạt động tổng quan - IAI_HACKATHON

Dự án IAI HACKATHON tập trung vào việc tối ưu hóa thuật toán nhân ma trận (Matrix Multiplication) bằng cách sử dụng công nghệ lập trình song song CUDA.

## 1. Kiến trúc tổng thể
Dự án bao gồm một thành phần cốt lõi nằm trong thư mục `IAI_HACKATHON_2025/`, trong đó file nguồn chính là `eg.cpp` (chứa mã nguồn CUDA C++).

Chương trình triển khai thuật toán nhân ma trận tối ưu bằng kỹ thuật **Tiling** kết hợp với **Shared Memory** (bộ nhớ dùng chung) trên GPU. Kỹ thuật này giúp giảm thiểu đáng kể độ trễ truy cập bộ nhớ toàn cục (Global Memory) bằng cách tải từng phần nhỏ (tile) của ma trận vào bộ nhớ dùng chung tốc độ cao của mỗi Block xử lý trước khi tiến hành tính toán.

## 2. Luồng hoạt động chính
Luồng hoạt động của dự án xoay quanh việc thiết lập môi trường trên Host (CPU) và gọi hàm thực thi trên Device (GPU). Mặc dù đoạn mã hiện tại tập trung vào việc định nghĩa hạt nhân (Kernel), luồng tổng thể sẽ diễn ra theo các bước sau:

1. **Khởi tạo và cấp phát (Host):**
   - CPU (Host) cấp phát bộ nhớ và khởi tạo dữ liệu cho hai ma trận đầu vào A và B.
   - Cấp phát vùng nhớ tương ứng trên GPU (Device) cho các ma trận A, B và ma trận kết quả C.
   - Dữ liệu của ma trận A và B được sao chép từ bộ nhớ CPU sang bộ nhớ GPU.

2. **Cấu hình thực thi:**
   - Cấu hình Grid và Block: Quá trình tính toán được chia nhỏ dựa trên hằng số `TILE_SIZE` (mặc định là 16). GPU sẽ tổ chức các luồng (threads) thành các Block kích thước 16x16, và một Grid chứa nhiều Block bao phủ toàn bộ kích thước ma trận.

3. **Thực thi trên GPU (Kernel Launch - Device):**
   - Hàm kernel `matMulShared` được CPU kích hoạt để chạy trên GPU. Hàng nghìn luồng trên GPU sẽ thực thi hàm này một cách song song.
   - Thuật toán thực hiện chia ma trận thành các khối nhỏ (tiles). Các luồng trong cùng một block sẽ hợp tác nạp dữ liệu từ Global Memory vào Shared Memory, đồng bộ hóa (`__syncthreads()`), tính toán một phần kết quả, và lặp lại cho đến khi hoàn thành tính toán cho toàn bộ chiều rộng ma trận.
   - Ghi kết quả từng phần tử tương ứng của ma trận C vào bộ nhớ toàn cục.

4. **Lấy kết quả (Host):**
   - Sau khi tất cả các luồng trên GPU hoàn tất, CPU tiến hành sao chép ma trận kết quả C từ bộ nhớ GPU về bộ nhớ CPU để sử dụng.
   - Giải phóng tài nguyên bộ nhớ đã cấp phát trên cả CPU và GPU.

---
*Để xem phân tích kỹ thuật chi tiết bên trong hàm kernel tính toán của GPU, vui lòng tham khảo file `IAI_HACKATHON_2025/IAI_HACKATHON_2025.md`.*
