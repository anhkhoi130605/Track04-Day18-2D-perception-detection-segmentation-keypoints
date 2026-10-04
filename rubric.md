# Lab Ngày 18 — Thang điểm (100 điểm lõi + 20 bonus)

Bám sát bài giảng *2D Perception: Detection · Segmentation · Keypoints*. Mỗi tiêu chí được chấm bằng **output trong
notebook đã chạy** và file `submission/ket_qua.json` do ô cuối notebook tạo ra.

| # | Mục | Tiêu chí | Điểm |
|---|---|---|---:|
| 1 | 1B | `box_iou` qua bộ kiểm tra (khớp `torchvision.ops.box_iou`) | 8 |
| 2 | 1B | `nms` qua bộ kiểm tra (khớp `torchvision.ops.nms` ở ngưỡng 0.3 / 0.5 / 0.7) | 10 |
| 3 | 1B | `batched_nms` qua bộ kiểm tra (class-aware, khớp torchvision) | 6 |
| 4 | 1C | Bảng latency đủ 4 cấu hình: one-to-many + NMS và one-to-one, ở conf 0.25 và 0.001 | 5 |
| 5 | 2B | `mask_iou` qua bộ kiểm tra (nhân ma trận, ra đúng shape `(N, M)`) | 8 |
| 6 | 2B | `polygon_to_mask` và `mask_to_yolo_seg` qua bộ kiểm tra (round-trip IoU > 0.95) | 6 |
| 7 | 2C | `submission/autolabel/bus.txt` hợp lệ: đúng định dạng YOLO-seg, toạ độ trong [0, 1], polygon khớp mask SAM | 5 |
| 8 | 3B | `oks` qua bộ kiểm tra (khớp `kpt_iou` của Ultralytics, xử lý đúng keypoint không có nhãn) | 8 |
| 9 | 3C | `joint_angle` qua bộ kiểm tra | 4 |
| 10 | 4A | `FLIP_IDX` đúng theo quy ước giải phẫu | 4 |
| 11 | 4B | Fine-tune YOLO26n-pose trên tiger-pose chạy xong trên GPU (40 epoch, imgsz 640) và có bảng mAP | 6 |
| 12 | Q1–Q12 | 12 câu hỏi, mỗi câu 2 điểm: trả lời đúng trọng tâm, có dẫn số liệu từ lần chạy của chính bạn khi câu hỏi yêu cầu | 24 |
| 13 | Toàn bài | Tái lập được: `Restart session and run all` không ô nào lỗi, notebook nộp còn nguyên output | 6 |
|   |   | **Tổng điểm lõi** | **100** |

**Mượn phao.** Hàm nào đi tiếp bằng `gate(..., lifeline=True)` thì tiêu chí của riêng hàm đó được 0 điểm; các tiêu chí khác
không bị ảnh hưởng. Trạng thái này được ghi trong `ket_qua.json` (`"lifeline"`).

**Q11 (phân tích lỗi)** được chấm theo lập luận: phải nêu được hai kiểu lỗi quan sát thấy trên ảnh val của chính bạn và một
cách khắc phục cụ thể cho từng kiểu. Trả lời chung chung không gắn với ảnh nào thì không được điểm.

## Bonus (tuỳ chọn, 20 điểm)

| Tiêu chí | Điểm |
|---|---:|
| 1D — `average_precision` qua bộ kiểm tra và vẽ được đường PR của ví dụ trên slide | 5 |
| 4C — bảng so sánh hai model (`flip_idx` giải phẫu và đồng nhất) trên val gốc và val lật gương, kèm một đoạn giải thích metric nào đã che lỗi | 10 |
| Một bài tập về nhà ở cuối notebook (auto-label + fine-tune `yolo26n-seg`, hoặc export ONNX và đo latency), có báo cáo ngắn trong `submission/` | 5 |
| **Tổng bonus** | **20** |

Không làm bonus không làm giảm điểm lõi.

## Nộp bài

**Không mở PR. Nộp một URL GitHub public vào ô LMS Ngày 18.**

1. Đẩy bài lên `<username-của-bạn>/Track04-Day18-2D-perception-detection-segmentation-keypoints` (fork hoặc repo mới, **public**).
2. Repo phải có:
   - `lab_2d_perception_student.ipynb` đã chạy hết, **còn nguyên output**;
   - `submission/ket_qua.json`;
   - `submission/autolabel/bus.txt`;
   - (tuỳ chọn) báo cáo bonus trong `submission/`.
3. Dán URL repo vào ô LMS. **Giữ repo public cho đến khi có điểm.** Repo private = 0 điểm.
