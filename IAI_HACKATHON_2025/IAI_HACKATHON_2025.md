# Phân tích chi tiết luồng hoạt động - Hàm Kernel `matMulShared`

File này đi sâu vào việc phân tích luồng hoạt động mã nguồn CUDA C++ trong file `eg.cpp`, cụ thể là hàm kernel `matMulShared`. Hàm này áp dụng kỹ thuật **Tiling** kết hợp **Shared Memory** để tăng tốc độ nhân hai ma trận vuông kích thước `width x width`.

## 1. Mục đích của Hàm
```cpp
__global__ void matMulShared(const float* A, const float* B, float* C, int width)
```
- Từ khóa `__global__` chỉ định đây là một hàm Kernel: được CPU (Host) gọi nhưng thực thi hoàn toàn song song trên GPU (Device).
- Nhận đầu vào là con trỏ mảng 1 chiều biểu diễn ma trận `A`, `B` và ghi kết quả vào ma trận `C`. Biến `width` là chiều rộng (và cũng là chiều cao) của ma trận vuông.

## 2. Luồng xử lý chi tiết trong Kernel

### Bước 1: Khai báo Bộ nhớ dùng chung (Shared Memory)
```cpp
__shared__ float tileA[TILE_SIZE][TILE_SIZE];
__shared__ float tileB[TILE_SIZE][TILE_SIZE];
```
- Khai báo 2 ma trận nhỏ `tileA` và `tileB` lưu trong **Shared Memory** của một Block.
- Shared memory có tốc độ truy xuất cực nhanh (gần bằng thanh ghi). Mọi thread trong cùng 1 Block đều có thể truy xuất 2 mảng này. `TILE_SIZE` được định nghĩa là 16.

### Bước 2: Xác định Tọa độ Thread
```cpp
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
```
Mỗi luồng (thread) chịu trách nhiệm tính toán một phần tử duy nhất `C[row][col]`:
- `row`: Chỉ số hàng toàn cục của phần tử cần tính.
- `col`: Chỉ số cột toàn cục của phần tử cần tính.
- `threadIdx.y`, `threadIdx.x`: Vị trí tương đối của thread trong 1 Block (từ `0` đến `TILE_SIZE-1`).

### Bước 3: Thuật toán Tiling - Duyệt qua các Tile
```cpp
float sum = 0.0f;
for (int t = 0; t < width / TILE_SIZE; ++t) { ... }
```
Thay vì nạp toàn bộ hàng của ma trận A và cột của ma trận B từ Global Memory (rất chậm), quá trình được chia thành các bước (phases / tiles). Mỗi bước xử lý một phần nhỏ kích thước `TILE_SIZE x TILE_SIZE`.

**Bên trong vòng lặp:**

1. **Nạp dữ liệu vào Shared Memory:**
   ```cpp
   tileA[threadIdx.y][threadIdx.x] = A[row * width + (t * TILE_SIZE + threadIdx.x)];
   tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
   ```
   Các thread trong Block hợp tác với nhau, mỗi thread tải đúng 1 phần tử của ma trận A và 1 phần tử của ma trận B từ Global Memory vào mảng Shared Memory `tileA` và `tileB`.

2. **Đồng bộ hóa (Lần 1):**
   ```cpp
   __syncthreads();
   ```
   Đảm bảo tất cả các thread trong Block đã nạp xong dữ liệu trước khi bắt đầu tính toán. Nếu không đồng bộ, một thread có thể đọc dữ liệu trong Shared Memory khi thread khác chưa kịp nạp xong.

3. **Tính toán:**
   ```cpp
   for (int i = 0; i < TILE_SIZE; ++i)
       sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
   ```
   Thực hiện nhân vô hướng một hàng của `tileA` và một cột của `tileB`. Kết quả cộng dồn vào biến tạm `sum` nằm trong thanh ghi (register) của thread.

4. **Đồng bộ hóa (Lần 2):**
   ```cpp
   __syncthreads();
   ```
   Đảm bảo toàn bộ các thread trong Block đã hoàn thành việc tính toán với `tileA` và `tileB` ở vòng lặp hiện tại, trước khi ghi đè dữ liệu mới cho vòng lặp tiếp theo.

### Bước 4: Ghi kết quả
```cpp
if (row < width && col < width)
    C[row * width + col] = sum;
```
Sau khi hoàn thành vòng lặp lớn, biến `sum` chứa giá trị chính xác của phần tử `C[row][col]`.
Điều kiện `if` đảm bảo các thread nằm ngoài biên của ma trận (trong trường hợp `width` không chia hết cho `TILE_SIZE`) sẽ không ghi kết quả rác hoặc gây lỗi truy cập bộ nhớ.
