# MLP:
- *Refs*:
    - https://www.youtube.com/watch?v=A83BbHFoKb8
- *Cấu trúc*:
    - Input layers:
        - Lớp đầu tiên, nhận dữ liệu từ bên ngoài
        - Mỗi nơ-ron trong lớp đầu vào *đại diện* cho một đặc trưng (feature)
    - Hidden layers:
        - Lớp trung gian giữa input layers và output layers
        - Bao gồm các nơ-ron hoạt động như một đơn vị tính toán, nhận đầu vào từ lớp trước, tính toán một giá trị mới, và truyền giá trị đó cho lớp kế tiếp
        - Activation function:
            - Áp dụng sau mỗi nơ-ron để giúp mô hình học được các quan hệ phi tuyến 
            - Các hàm kích hoạt phổ biến:
                - ReLU
                - Sigmoid
                - Tanh
                - ...
    - Output layers:
        - Lớp cuối cùng -> tạo ra kết quả của mạng nơ-ron
        - Lớp đầu ra có số lượng nơ-ron phù hợp với từng loại bài toán:
            - Với bài toán phân loại (classification) -> số nơ-ron tương ứng với số lớp phân loại
            - Với bài toán hồi quy (regression) -> lớp đầu ra có thể chỉ có một nơ-ron
        - Hàm kích hoạt trong lớp đầu ra tùy thuộc vào từng loại bài toán:
            - *Sigmoid* cho phân loại nhị phân
            - *Softmax* cho phân loại đa lớp
            - Không có hàm kích hoạt hoặc *linear activation* cho bài toán hồi quy
- *Quy trình hoạt động MLP*:
    - Forward Propagation (Lan truyền tiến):

        - Dữ liệu đầu vào sẽ đi qua từng lớp, được tính toán qua các trọng số và bias, và đi qua các hàm kích hoạt.

        - Mỗi lớp sẽ tính toán đầu ra của mình và truyền sang lớp tiếp theo cho đến khi đạt được lớp đầu ra.

        - Mục tiêu là giảm thiểu sự khác biệt giữa kết quả đầu ra và giá trị mục tiêu thông qua việc cập nhật trọng số.

    - Backpropagation (Lan truyền ngược):

        - Sau khi tính toán được lỗi (error) giữa giá trị dự đoán và giá trị thực tế, ta sử dụng thuật toán backpropagation để tính toán đạo hàm của hàm lỗi đối với trọng số và bias của từng lớp.

        - Sau đó, sử dụng Gradient Descent để cập nhật trọng số và bias nhằm giảm thiểu lỗi trong các vòng lặp tiếp theo.

    - Cập nhật trọng số:

        - Trọng số sẽ được cập nhật thông qua thuật toán gradient descent, sử dụng đạo hàm của hàm lỗi và learning rate (tốc độ học) để điều chỉnh trọng số sao cho mô hình học được các mối quan hệ trong dữ liệu.