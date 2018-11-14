# Bước 1.Tạo thư mục như mẫu sau để chạy chương trình

```bash
|---db/
    |---db1/
|---exp/
    |---knn/
        |---db1/
    |---svmlinear/
        |---db1/
|---features/
    |---vgg16_fc2/
|---images/
|---1mainPrepareTraining.py
|---2mainTraining.py
|---3mainShowingResult.py
|---clustering.py
|---extract_features.py
|---generate_db.py
```
- Chương cần ảnh trong mục images để chạy!
- File kết quả của tập ảnh calteach256 đã được đính kèm trong folder [exp](https://github.com/thienminhaquarius/imagesClassification/tree/master/exp). Đến Bước 4 để xem file kết quả mà không cần chạy tập ảnh calteach256
- Chương trình chạy trên môi trường python 3.6.0, xem them cài các thư viện nếu chưa cài đặt [docs](https://github.com/thienminhaquarius/imagesClassification/blob/master/docs.pdf)


# Bước 2. Chạy 1mainPrepareTraining.py để Generate data path và features extraction
*   
    - Nhấp chuột phải vào file 1mainPrepareTraining.py và chọn "Edit with IDLE" (phía dưới Open).
    - Giao diện code python hiện ra, nhất F5 để chạy chương trình. 
# Bước 3. Chạy 2mainTraining.py data để tạo model và kết quả (sau khi đã chạy xong ở Bước 2)
* Lưu ý: Bước này sẽ ghi đè file kết quả của tập ảnh calteach256 có sẵn trong [exp](https://github.com/thienminhaquarius/imagesClassification/tree/master/exp)
    - Nhấp chuột phải vào file 2mainTraining.py và chọn "Edit with IDLE" (phía dưới Open).
    - Giao diện code python hiện ra, nhất F5 để chạy chương trình.
# Bước 4. Chạy 3mainShowingResult.py để xem kết quả. 

- Chạy file này để xem kết quả có sẵn trong folder [exp](https://github.com/thienminhaquarius/imagesClassification/tree/master/exp) hoặc sau khi đã chạy xong Bước 3 để xem kết quả trên máy tính của bạn!.  
    - Nhấp chuột phải vào file 3mainShowingResult.py và chọn "Edit with IDLE" (phía dưới Open).
    - Giao diện code python hiện ra, nhất F5 để chạy chương trình.

# Video chạy Demo tập ảnh nhỏ của chương trình.
* [https://www.youtube.com/watch?v=LEoeGjVxPw0](https://www.youtube.com/watch?v=LEoeGjVxPw0)

# Tài liệu tham khảo
*   Deep Learning for Computer Vision with Python - Starter Bundle - Dr. Adrian Rosebrock
*   [Machine learning 101](https://medium.com/machine-learning-101)
