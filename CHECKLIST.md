# Checklist Lab Day 10

## Môi trường và dữ liệu
- [x] Colab T4, `requirements-colab.txt`; gỡ `torchao` 0.10 có sẵn (xung đột với peft 0.21.2)
- [x] `python -m pytest -q`: 68 passed, 1 skipped
- [x] Dữ liệu: `prepare_data.py` (Kaggle CLI, hoặc `kagglehub` ẩn danh rồi `--source`), `dataset_version` = `d10-21e4070301`
- [x] Số ảnh và box từng split khớp bảng của đề; không phiên quay nào nằm ở hai split

## Nhãn và E0
- [x] `yolo_to_box` / `objects_to_answer` (thang 0–1000, sắp theo y1, x1); sai số quy đổi ≤ 0.32 px
- [x] Đưa đáp án đúng qua trọng tài: đúng thang F1 = 1.0, lệch thang F1 = 0.0
- [x] `report/evidence/e0.json`: F1 0.086, parse_valid 0.836

## Huấn luyện E1–E3
- [x] Mask loss: 220 token ảnh; phần có nhãn giải mã ra đúng JSON + `<|im_end|>`
- [x] LoRA: 252 module (E1, E2) / 348 module (E3); projector mở huấn luyện = 109,105,152 tham số, fp32
- [x] ViT và projector **không** bị lượng tử hoá (`llm_int8_skip_modules=["model.visual", "lm_head"]`, có assert)
- [x] Lưu adapter + `projector.pt` riêng; nạp lại và kiểm tra |Δ| > 0
- [x] `e1.json` – `e3.json` có `meta` (trainable_params, peak_vram_gb, train_minutes)
- [x] E1: parse_valid 0.990 ≥ 0.95; F1(E1) − F1(E0) = 0.193 ≥ 0.10

## E4, E5
- [x] E4 `data_size` 0.1 / 1.0 (gốc E3): `e4_data_0.1.json`, `e4_data_1.0.json`; CI95 + paired bootstrap
- [x] E5: hàm reward + 11 test (`python -m pytest lab -q`), GRPO 60 bước có log
- [ ] E5: suy luận + `e5.json` + so sánh với SFT — không chạy (session bị ngắt, quyết định dừng ở GRPO + log)

## Báo cáo và nộp
- [x] `REPORT.md`: bảng E0–E4 và 10 câu hỏi bằng số thật
- [x] Ảnh minh hoạ: `pred_examples.png` (3 đúng + 3 sai), `per_class_errors.png`, `data_samples.png`, `loss_*.png`
- [x] Điền họ tên trong `REPORT.md`
- [x] Không sửa `scripts/` và `tests/`; không commit `data/`, `work/`, `.env`, trọng số
- [x] `python scripts/check_submission.py`: 0 FAIL
- [x] Repo GitHub công khai, ≥ 5 commit
