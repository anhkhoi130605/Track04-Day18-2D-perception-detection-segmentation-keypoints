# Lab 18 — 2D Perception: Detection · Segmentation · Keypoints (Track 4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinUni-AI20k/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb)

> 🏭 Camera ở cổng nhà máy cần biết: **có bao nhiêu người**, **ai không đội mũ bảo hộ**, và **có ai vừa ngã**.
> Bạn dùng một model hay ba — và output của mỗi model trông như thế nào?

Đó là câu hỏi mở đầu bài giảng. Trong 120 phút của lab này, bạn tự tay dựng từng mảnh của câu trả lời:

```
ảnh ─▶ detector ─▶ NMS (bạn tự viết) / head one-to-one ─▶ đếm người                (Phần 1)
ảnh ─▶ YOLO26 box ─▶ SAM 2.1 ─▶ mask ─▶ polygon (bạn tự viết) ─▶ file nhãn YOLO-seg   (Phần 2)
ảnh ─▶ YOLO26-pose ─▶ 17 keypoint ─▶ góc khớp (bạn tự viết) ─▶ luật "ngã"            (Phần 3)
dữ liệu custom ─▶ sửa flip_idx ─▶ fine-tune ─▶ chấm bằng OKS (bạn tự viết) ─▶ xem lỗi  (Phần 4)
```

Toàn bộ lab nằm trong **một notebook**: [`lab_2d_perception_student.ipynb`](lab_2d_perception_student.ipynb).
Không cần API key, không cần tự chuẩn bị dữ liệu: notebook tự tải weights và dataset (khoảng 0,8 GB).

---

## Bắt đầu nhanh

1. Bấm nút **Open in Colab** ở trên, rồi `File → Save a copy in Drive` để có bản của riêng bạn.
   (Nếu nút không mở được, tải file `.ipynb` về và dùng `File → Upload notebook` trên [Colab](https://colab.research.google.com).)
2. `Runtime → Change runtime type → T4 GPU`.
3. Chạy lần lượt 5 ô của **Phần 0**. Ô thứ tư tải toàn bộ weights và dataset; hãy chạy nó trong giờ nghỉ.
4. Làm từ Phần 1 đến Phần 4 theo thứ tự. Các phần sau dùng lại hàm bạn viết ở phần trước.

Notebook cố định `ultralytics==8.4.171`. Không có GPU thì notebook vẫn chạy, nhưng Phần 4 tự giảm xuống 3 epoch và kết quả
fine-tune sẽ không dùng được để nộp.

---

## Lộ trình 120 phút

| Phần | Nội dung | Bạn tự code | Mảnh ghép cho camera cổng | Phút |
|---|---|---|---|---:|
| 0 · Setup | GPU, cài đặt, tải sẵn weights và dataset | — | — | 10 |
| 1 · Detection | Faster R-CNN và YOLO26 · IoU, NMS · one-to-many + NMS so với one-to-one · latency | `box_iou`, `nms`, `batched_nms`, ⭐ `average_precision` | Đếm người | 35 |
| 2 · Segmentation | Semantic so với instance · mask IoU · SAM 2.1 → auto-label | `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg` | Tạo nhãn cho class mới | 30 |
| 3 · Keypoints | Keypoint R-CNN so với YOLO26-pose · OKS · góc khớp | `oks`, `joint_angle` | Phát hiện người ngã | 25 |
| 4 · Fine-tune | tiger-pose (12 keypoint) · `flip_idx` · phân tích lỗi | `FLIP_IDX` | Dạy model một đối tượng mới | 20 |

---

## Cách làm việc với notebook

| Ký hiệu | Ý nghĩa |
|---|---|
| 🔮 **Dự đoán** | Tự nhẩm kết quả *trước khi* chạy ô. Không chấm điểm, nhưng là lúc bạn học được nhiều nhất. |
| 🧩 **TODO** | Điền code vào chỗ có `...`. Mỗi chỗ chỉ một dòng. Chạy ô là có phản hồi ngay bên dưới. |
| ✅ / ❌ | Kết quả của bộ kiểm tra. Dòng ❌ nói rõ phép thử nào hỏng và nên xem lại bước nào. |
| 💡 **Gợi ý** | Bấm để mở. Kẹt quá 3 phút thì mở. |
| 🚦 **Chốt** | Ô `gate("1B")` ở cuối mỗi mục TODO. Còn hàm chưa đạt thì notebook dừng ở đó với một thông báo ngắn. |
| 🛟 **Phao** | Hết giờ của mục mà chưa xong: sửa thành `gate("1B", lifeline=True)` để mượn bản thư viện và đi tiếp. Hàm mượn phao không được tính điểm; làm xong sau thì chạy lại ô TODO, trạng thái tự đổi về ✅. |
| ❓ **Câu hỏi** | 12 câu (Q1–Q12). Trả lời 2–4 câu vào chuỗi `Q1 = """ ... """` ngay bên dưới câu hỏi. |
| ⭐ | Nâng cao, có điểm bonus. |

Một ô TODO trông như thế này. Bạn chỉ thay các dấu `...`:

```python
def box_iou(a, b):
    # Bước 1 — diện tích từng box = (x2 − x1) · (y2 − y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = ...                                    # 🧩
    ...
check_box_iou()
```

Và phản hồi khi còn sai:

```
❌ box_iou — hai box rời nhau
   ↳ A và C không chạm nhau nên IoU phải = 0, bạn ra 1.0000. Ở Bước 3, (rb − lt) âm theo cả hai chiều thì tích lại dương → thêm .clamp(min=0).
```

---

## Hướng dẫn từng phần

Mỗi mục dưới đây nêu việc cần làm và **dấu hiệu cho biết bạn đã làm đúng**. Các con số là kết quả khi chạy với
`ultralytics==8.4.171`; riêng số đo thời gian và mAP sẽ khác đôi chút giữa các máy.

### Phần 0 — Setup (10')

Chạy 5 ô. Bạn đã sẵn sàng khi thấy dòng `device = cuda`, dòng `🔧 Sẵn sàng: ...`, danh sách 9 dấu `✓`, và hai ảnh
`bus.jpg`, `zidane.jpg`.

### Phần 1 — Object Detection (35')

| Mục | Việc cần làm | Bạn đã làm đúng khi |
|---|---|---|
| **1A** (7') | Chạy 3 ô, đọc shape của output. Trả lời **Q1**. | Faster R-CNN in ra `boxes / labels / scores`; YOLO26n in ra `(1, 84, 8400)` cho ảnh 640×640 và `(1, 84, 6300)` cho `bus.jpg`; hình cuối có một box xanh đúng và một box đỏ lệch. |
| **1B** (15') | Điền `box_iou` (4 chỗ), `nms` (2 chỗ), `batched_nms` (2 chỗ). | Mỗi ô in `✅ ... đạt`. Với ba box ví dụ A, B, C: IoU(A, B) = 0.3333, NMS ngưỡng 0.3 giữ A và C, ngưỡng 0.5 giữ cả ba. Ô `gate("1B")` in `✅ Mục 1B xong`. |
| **1C** (8') | Chạy 2 ô. Trả lời **Q2**, **Q3**. | Ba hình lần lượt có **51 → 5 → 5** box; dòng `NMS của bạn giữ 5 box · NMS của Ultralytics giữ 5 box ✅ khớp`; camera đếm được 4 người, 1 xe buýt; bảng latency đủ 4 dòng. |
| **1D** ⭐ | Điền `average_precision` (3 chỗ). | `✅ average_precision đạt` và hình đường PR ghi AP ≈ 0.535, đúng ví dụ trên slide. |

### Phần 2 — Segmentation (30')

| Mục | Việc cần làm | Bạn đã làm đúng khi |
|---|---|---|
| **2A** (8') | Chạy 2 ô. Trả lời **Q4**. | Bản đồ semantic cho ra **7 vùng liên thông** của class `person` trong khi ảnh chỉ có 4 người; phần lớn pixel của xe buýt bị gán class `train`. Cả hai đều là kết quả đúng của bài, không phải lỗi của bạn. |
| **2B** (10') | Điền `mask_iou` (3 chỗ), `polygon_to_mask` (1 chỗ), `mask_to_yolo_seg` (2 chỗ). Trả lời **Q5**. | Ba dòng `✅ ... đạt`; ví dụ hai hình vuông ra IoU = 0.1429; bảng ghép cặp YOLO26n-seg với Mask R-CNN có 5 dòng, mask IoU khoảng 0.84–0.94. |
| **2C** (12') | Chạy 2 ô. Trả lời **Q6**. | Điểm A cho mask cả người (IoU ≈ 0.99), điểm B cho một mảnh nhỏ (IoU ≈ 0.00); file `autolabel/bus.txt` có 5 dòng; hình vẽ lại polygon bám sát người và xe; dòng `✅ autolabel/bus.txt hợp lệ`. |

### Phần 3 — Keypoints và Pose (25')

| Mục | Việc cần làm | Bạn đã làm đúng khi |
|---|---|---|
| **3A** (8') | Chạy 2 ô, rồi đổi `KP_THR = -100` và chạy lại ô thứ hai. Trả lời **Q7**. | Keypoint R-CNN ra `(2, 17, 3)`, YOLO26n-pose ra `(2, 17, 2)`; khi bỏ ngưỡng, Keypoint R-CNN vẽ cả đầu gối và mắt cá dù chúng nằm ngoài ảnh. |
| **3B** (10') | Điền `oks` (3 chỗ). Chạy ô thí nghiệm. Trả lời **Q8**. | `✅ oks đạt`; đồ thị có ba đường cong theo kích thước người; dòng `Lệch 8 px: mắt còn 0.53, hông còn 0.97`. |
| **3C** (7') | Điền `joint_angle` (2 chỗ). Chạy ô áp dụng. Trả lời **Q9**. | `✅ joint_angle đạt`; trên ảnh gốc thân nghiêng khoảng 1° (`đứng`), trên ảnh xoay 90° thân nghiêng 83–94° (`🚨 NGÃ?`) và model chỉ tìm được 3 người. |

### Phần 4 — Fine-tune YOLO26n-pose (20')

| Mục | Việc cần làm | Bạn đã làm đúng khi |
|---|---|---|
| **4A** (6') | Chạy ô thống kê hướng quay. Sửa `FLIP_IDX`. Trả lời **Q10**. | Train 210/0 và val 53/0 (mọi con hổ đều quay phải); `✅ FLIP_IDX đạt`; hình ba ô, ở ô thứ ba các chân phía camera đổi sang màu xanh. |
| **4B** (12') | Chạy ô train (40 epoch), ô đánh giá, ô phân tích lỗi. Trả lời **Q11**, **Q12**. | Bảng kết quả có Pose mAP50 trên 0.9; hình 6 ảnh val tệ nhất và biểu đồ sai số theo keypoint hiện ra. Q11 phải dựa trên chính các hình này. |
| **4C** ⭐ | Đặt `RUN_4C = True` (và `TRAIN_IDENTITY = True` nếu muốn so sánh hai model). | Bảng mAP trên val gốc và val lật gương. |

Ô cuối cùng `final_report()` in checklist. Mọi dòng bắt buộc phải là ✅ trước khi nộp.

---

## Chấm điểm và nộp bài

Xem [`rubric.md`](rubric.md) (100 điểm lõi + 20 bonus).

1. `Runtime → Restart session and run all`, chờ chạy hết, không ô nào báo lỗi.
2. Chạy ô cuối. Ô này tạo thư mục `submission/` và file `submission.zip`, gồm `ket_qua.json` và `autolabel/bus.txt`.
3. Tải về **notebook đã chạy** (`File → Download → Download .ipynb`) và **`submission.zip`** (biểu tượng 📁 ở cột trái của Colab), rồi giải nén.
4. Đẩy notebook và thư mục `submission/` lên repo GitHub **public** của bạn:
   `<username-của-bạn>/Track04-Day18-2D-perception-detection-segmentation-keypoints` (fork hoặc repo mới).
5. Dán URL repo vào ô LMS Ngày 18. **Không mở PR.** Giữ repo public cho đến khi có điểm; repo private = 0 điểm.

---

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `⚠️ Không thấy GPU` ở Phần 0 | `Runtime → Change runtime type → T4 GPU`, rồi chạy lại từ đầu. |
| `⛔ Mục 1B chưa xong: ...` | Đây là ô chốt, không phải lỗi hệ thống. Kéo lên ô TODO, đọc dòng ❌, mở 💡 gợi ý. Hết giờ thì dùng phao. |
| `❌ ... — còn 2 chỗ ... chưa điền` | Hàm vẫn còn dấu `...`. Điền hết rồi chạy lại chính ô đó. |
| Sửa code rồi mà vẫn ❌ | Bạn phải **chạy lại ô TODO** (Shift + Enter) thì hàm mới được định nghĩa lại. |
| Ô tải weights báo lỗi mạng | Chạy lại ô đó; các file đã tải xong sẽ không tải lại. |
| `CUDA out of memory` ở 4B | `Runtime → Restart session and run all`. Nếu vẫn lỗi, đổi `batch=16` thành `batch=8` trong ô train. |
| Colab ngắt kết nối giữa chừng | Kết nối lại và `Run all`. Các câu trả lời Q1–Q12 nằm trong notebook nên không mất, miễn là bạn đang làm trên bản đã lưu vào Drive. |

---

## Bài tập về nhà (chọn 1, có điểm bonus)

1. **Tập val có nói thật không?** Chạy thí nghiệm 4C với cả hai model, thử thêm `kpt_oks_sigmas` tự ước lượng trong YAML,
   và viết một đoạn: metric nào đã che lỗi `flip_idx`, bạn sẽ thiết kế tập val thế nào.
2. **Auto-label cho bài toán của bạn.** Gán nhãn tự động 20–30 ảnh trong lĩnh vực của bạn bằng YOLO26 + SAM 2.1 (mục 2C),
   sửa tay các ca sai, fine-tune `yolo26n-seg.pt` và báo cáo mask mAP.
3. **Đo cả pipeline.** Export `yolo26n.pt` sang ONNX (`model.export(format="onnx")`), đo latency trên CPU của head
   one-to-many + NMS và head one-to-one ở conf 0.001 và 0.25.

---

## Lab này ứng với phần nào của bài giảng

| Mục lab | Phần trong slide |
|---|---|
| 1A | *Bốn quy ước box* · *Faster R-CNN: hai giai đoạn* · *One-stage: dự đoán dày trên đa tỷ lệ* |
| 1B | *IoU và các biến thể* · *NMS: dọn duplicate của one-to-many* |
| 1C | *Ba vấn đề của NMS* · *Hai trường phái hội tụ ở one-to-one* · *Triển khai: đo cả pipeline* |
| 1D | *Từ đường PR đến AP* · *mAP@[.5:.95] và bảng COCO* |
| 2A | *Semantic segmentation* · *Panoptic: things + stuff* |
| 2B | *Loss và metric cho segmentation* · *Mask R-CNN* · *YOLO-seg: prototypes × coefficients* |
| 2C | *SAM 1 → 2 → 3* · *Auto-labeling: model lớn dạy model nhỏ* |
| 3A | *Bốn cách biểu diễn output keypoint* · *Ba kiến trúc pose nhiều người* |
| 3B | *OKS: IoU của keypoint* |
| 3C | *Quay lại câu hỏi: camera cổng nhà máy* |
| 4A, 4C | *Bẫy flip_idx: augmentation phải hiểu nhãn* · *Bốn cái bẫy khi đọc mAP* |
| 4B | *Vòng lặp phân tích lỗi* |
