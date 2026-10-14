# Báo cáo lỗi và lưu ý kỹ thuật

Những bẫy gặp khi chạy lab với đúng các phiên bản đã ghim: transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2, trên Colab T4.

## 1. `llm_int8_skip_modules=["visual", ...]` không giữ ViT ở fp16

- **Triệu chứng:**
  - E0 và E1 chạy bình thường.
  - E2 lỗi ngay bước đầu với `ValueError: Attempting to unscale FP16 gradients`.
  - Số tham số projector khi mở huấn luyện là 54,579,200, không phải 109,105,152.
- **Nguyên nhân:**
  - transformers 5.19 so khớp skip module bằng `re.match(pattern, full_name)` (khớp từ **đầu** tên) hoặc `full_name.endswith(pattern)`.
  - Tên thật là `model.visual.blocks.N…` và `model.visual.merger…`, nên `"visual"` không khớp cái nào và 104 Linear của ViT/projector bị nén 4-bit.
  - Params4bit lưu 2 trọng số mỗi byte, nên `numel()` chỉ bằng khoảng một nửa.
- **Cách sửa:**
  - Dùng `["model.visual", "lm_head"]`.
  - Assert ngay sau khi nạp: không có `Linear4bit` nào trong `visual`, đúng 252 `Linear4bit` (LLM).
  - Assert projector mở huấn luyện = 109,105,152 tham số.
- **Hệ quả:** phải chạy lại E0–E3. Công cụ chấm không phát hiện được lỗi này, vì E0/E1 vẫn đạt mọi tiêu chí.

## 2. Chọn module LoRA bằng hậu tố tên

- Danh sách `["q_proj", …, "qkv", "linear_fc1", "linear_fc2"]` bị sai theo hai cách:
  - Khớp nhầm cả `visual.merger.linear_fc1/2` và `deepstack_merger_list.N.linear_fc1/2` (LoRA gắn lên projector).
  - Bỏ sót `attn.proj` của ViT.
- Cách đúng: regex theo tên đầy đủ (peft dùng `re.fullmatch`).
  - LLM: `.*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)`
  - ViT: `.*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))`
  - Kiểm tra: 252 / 348 module, không module nào thuộc projector.

## 3. torchao 0.10 có sẵn trên Colab làm peft lỗi ở E3

- `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
- Lỗi chỉ xảy ra khi gắn LoRA lên `nn.Linear` thường (ViT ở E3). Với `Linear4bit`, bitsandbytes được xử lý trước nên peft không gọi tới phần kiểm tra torchao.
- Sửa: `pip uninstall -y torchao` (lab không dùng torchao).

## 4. Projector phải được huấn luyện ở fp32 và lưu riêng

- Với autocast fp16 + `GradScaler`, tham số huấn luyện phải ở fp32 (master weights), nếu không sẽ gặp lỗi `unscale FP16 gradients`.
- LoRA của peft đã tự ở fp32; projector phải tự `.float()`.
- `PeftModel.save_pretrained` chỉ lưu LoRA → lưu `projector.pt` riêng. Khi nạp lại để suy luận, in |Δ| so với projector gốc để chắc không phải bản chưa huấn luyện; nếu quên, E2/E3 sẽ trông giống hệt E1.

## 5. trl 1.9.2 (E5 GRPO)

- `bf16` mặc định là `True` khi không đặt `fp16`. T4 không có bf16 nhanh → đặt `fp16=True, bf16=False`.
- Với model lượng tử, trl ép tham số huấn luyện về **bf16**. Ở lr 5e-6, cập nhật nhỏ hơn độ phân giải của bf16 (khoảng 3.9e-3, tính tương đối) bị làm tròn mất → đưa LoRA về fp32 sau khi khởi tạo trainer.
- `remove_unused_columns=True` (mặc định) xoá cột ground truth trước khi gọi hàm reward → đặt `False`.
- `max_completion_length=128` nhỏ hơn đáp án của 142 ảnh train (tối đa 366 token) → bỏ các ảnh này khỏi dữ liệu GRPO, vì completion của chúng chắc chắn bị cắt và nhận reward 0 vô nghĩa.

## 6. Các lưu ý nhỏ

- `processor.apply_chat_template(..., padding=True)` gây cảnh báo ở mỗi batch trong transformers 5.19 → truyền qua `processor_kwargs={"padding": True}` (kết quả như nhau).
- Processor trả thêm `mm_token_type_ids`; `collate_fn` phải pad cả khoá này.
- Ảnh 640×360 được resize thành 640×352: patch 16, grid 22×40, gộp 2×2 → 220 token (không phải patch 14).
- Model gốc của Qwen3-VL có config bf16; trên T4 nạp với `dtype=torch.float16`.
- Colab mở notebook từ GitHub theo nhánh có thể dùng bản lưu tạm cũ → mở theo link commit cụ thể, hoặc **File → Upload notebook**.
- Dataset là công khai (MIT): `kagglehub.dataset_download` tải được không cần khoá Kaggle.
- Session Colab miễn phí bị ngắt nhiều lần → checkpoint mỗi 100 bước + sao lưu Drive; `train_minutes` cộng dồn qua các session.
