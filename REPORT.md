Họ tên: Trương Thị Lan Anh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Môi trường: Colab T4 16 GB, transformers 5.19.0, peft 0.21.2. LLM ở 4-bit NF4; ViT, projector và `lm_head` ở fp16. 300 bước, mỗi bước 2 ảnh × tích luỹ 4, lr 2e-4. Thời gian và VRAM lấy từ `meta` (`torch.cuda.max_memory_allocated`).

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.086 | [0.061, 0.113] | 0.92 | 0.107 | 0.032 | 0.178 |
| E1 | LoRA LLM | 33,030,144 | 6.35 | 40.9 | 0.279 | [0.243, 0.318] | 1.36 | 0.202 | 0.211 | 0.532 |
| E2 | + projector | 142,135,296 | 8.05 | 46.0 | 0.366 | [0.323, 0.411] | 0.97 | 0.192 | 0.316 | 0.600 |
| E3 | + LoRA ViT | 148,426,752 | 8.22 | 52.7 | 0.418 | [0.377, 0.464] | 0.97 | 0.284 | 0.340 | 0.690 |
| E4 | data_size = 0.1 (gốc E3) | 148,426,752 | 7.67 | 53.3 | 0.267 | [0.235, 0.303] | 1.73 | 0.208 | 0.184 | 0.547 |
| E4 | data_size = 1.0 (= E3) | 148,426,752 | 8.22 | 52.7 | 0.418 | [0.377, 0.464] | 0.97 | 0.284 | 0.340 | 0.690 |

Các chỉ số khác: `parse_valid_rate` của E0/E1/E2/E3 lần lượt là 0.836 / 0.990 / 1.000 / 1.000. Mean IoU của các cặp khớp là 0.601 / 0.674 / 0.675 / 0.697.

Ghi chú:
- `train_minutes` của E1 cộng dồn qua hai session Colab, vì session bị ngắt ở bước 100 và chạy tiếp từ checkpoint.
- Dòng `data_size = 1.0` chấm lại chính predictions của E3 (đề cho phép một trong hai giá trị trùng setup gốc), nên số liệu trùng với E3.

Ảnh minh hoạ:
- `report/figures/pred_examples.png`: 3 ảnh đúng và 3 ảnh sai của E3. Box GT màu xanh lá, dự đoán khớp màu đỏ, dự đoán sai màu cam.
- `report/figures/per_class_errors.png`: lỗi tiêu biểu của từng lớp, kèm output E0 trên cùng ảnh.
- `report/figures/data_samples.png`: nhãn gốc.
- `report/figures/loss_e*.png`: đường loss.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm bằng `named_parameters()` trên model đã nạp. Với trọng số 4-bit, mình lấy shape gốc từ `quant_state`, vì `numel()` của Params4bit chỉ bằng một nửa số tham số thật.

| Khối | Tham số |
|---|---|
| ViT (patch embed, pos embed, 24 block) | 306,242,560 |
| Projector `visual.merger` | 27,271,680 |
| Projector `visual.deepstack_merger_list` (DeepStack, 3 merger) | 81,833,472 |
| LLM `language_model` (36 lớp, gồm embedding; `lm_head` dùng chung trọng số embedding) | 4,022,468,096 |
| Tổng | ≈ 4.438B |

Mỗi thí nghiệm cập nhật các khối sau:
- **E1**: chỉ LoRA r=16 trên 252 module của LLM (36 lớp × q/k/v/o/gate/up/down), tổng 33,030,144 tham số. ViT và projector đóng băng.
- **E2**: E1 cộng thêm toàn bộ projector ở fp32. Tổng 142,135,296; chênh lệch so với E1 đúng bằng 109,105,152 = 27.27M + 81.83M.
- **E3**: E2 cộng thêm LoRA trên 96 module của ViT (24 block × `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`). LoRA tăng từ 33.0M lên 39.3M, tổng 148,426,752.

Mình chọn module LoRA bằng regex theo tên đầy đủ. Lọc theo đuôi tên như `linear_fc1` là không được, vì tên này cũng khớp `visual.merger.linear_fc1` và sẽ gắn LoRA nhầm lên projector.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

E0 có `parse_valid_rate` = **0.836**, tức 51/311 output không parse được thành JSON hợp lệ. Precision 0.120, recall 0.067, F1 0.086. Hai lỗi phổ biến nhất:

1. **JSON hỏng: 51 ảnh (16.4%).** Trọng tài tính các ảnh này là bỏ sót toàn bộ box. Mình chưa phân loại kỹ nguyên nhân bên trong 51 output này (bị cắt ở 256 token, hay viết thêm chữ thừa).
2. **Bỏ sót và sai lớp: recall chỉ 0.067** (39 TP trên 583 box).
   - 17 ảnh model trả `[]` dù ảnh có hư hại.
   - Model gán nhãn `crack` 201 lần nhưng F1 lớp crack chỉ 0.032 (precision 0.04): phần lớn những gì nó gọi là "crack" không khớp vết nứt thật.
   - Khi đã khớp, box khá đúng chỗ (mean IoU 0.601) và không box nào vượt thang 1000. Như vậy model gốc đã dùng đúng quy ước 0–1000; cái nó thiếu là khái niệm "hư hại mặt đường" theo cách bộ dữ liệu này gán nhãn.

Ví dụ output thô của `vlcsnap-00200` (box `manhole` đầu tiên là box bịa, đồng thời bỏ sót vết nứt):
```
GT : [{"label":"crack","bbox_2d":[352,642,534,728]},{"label":"manhole","bbox_2d":[894,756,1000,847]}]
RAW: [{"label": "manhole", "bbox_2d": [770, 640, 843, 686]}, {"label": "manhole", "bbox_2d": [909, 786, 998, 822]}]
```
Ví dụ `vlcsnap-00201`: GT có 1 manhole và 1 crack, nhưng output thô là `[]`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

**F1 về 0.** Mình kiểm chứng bằng cách đưa chính đáp án đúng của 276 ảnh val qua `scripts.score.evaluate`:

| Đáp án | Chấm với | F1 | mean IoU |
|---|---|---|---|
| 0–1000 | `--coord norm1000` | 1.000 | 0.989 |
| 0–1000 | `--coord pixel` | **0.000** | 0.000 |
| pixel 640×360 | `--coord pixel` | 1.000 | 1.000 |
| pixel 640×360 | `--coord norm1000` | **0.000** | 0.000 |

Vì sao F1 về 0:
- Khi chấm `pixel`, trọng tài đổi toạ độ theo công thức x·1000/640 và y·1000/360. Một đáp án 0–1000 bị hiểu như pixel nên bị kéo giãn 1.56 lần theo x và 2.78 lần theo y.
- Box của bộ này rất nhỏ (diện tích trung vị 0.4–1.35% khung hình), nên bị dịch hoặc giãn như vậy là IoU về 0. Không cặp nào vượt ngưỡng 0.5, dù JSON vẫn hợp lệ 100%.
- Đây chính là triệu chứng "JSON hợp lệ mà F1 ≈ 0" trong GUIDE. Sai thang **không làm giảm `parse_valid_rate`**, nên không thể phát hiện chỉ bằng tỉ lệ parse.

Vì sao Qwen3-VL dùng thang 0–1000:
- Processor đổi kích thước ảnh động. Ảnh 640×360 của bộ này được resize thành 640×352 (grid 22×40, patch 16) trước khi vào ViT. Model không thấy độ phân giải gốc, nên pixel tuyệt đối là đại lượng mơ hồ với nó.
- Thang tương đối không phụ thuộc kích thước ảnh, và là quy ước model đã học khi pretrain. Bằng chứng: E0 không có toạ độ nào > 1000, và mean IoU khi khớp là 0.601.

Mình còn kiểm tra quy đổi YOLO → 0–1000 → pixel trên toàn bộ train: sai số lớn nhất 0.32 px, đúng bằng sai số làm tròn 0.5/1000 × 640.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Đo trên 50 ảnh train: mỗi chuỗi trung bình **384 token**, chỉ **72 token (18.9%)** là câu trả lời có loss. Riêng **220 token ảnh** đã chiếm **57.2%** chuỗi. Ảnh 640×360 resize thành 640×352, grid 1×22×40 = 880 patch, gộp 2×2 thành 220 token. Phần còn lại là prompt cố định và các token template.

Cách mask:
- Mã hoá `[user]` với `add_generation_prompt=True`, lấy độ dài n. Kiểm tra n token này đúng là tiền tố của `[user, assistant]`.
- Gán `labels[:n] = -100`.
- Giải mã phần `labels != -100` ra đúng JSON đáp án + `<|im_end|>\n`. Token `<|im_end|>` cần được học để model biết dừng (`parse_valid_rate` của E1–E3 đạt 0.990–1.000).

Nếu tính loss trên cả prompt và ảnh:
- 220 vị trí ảnh đều là cùng một token giữ chỗ `<|image_pad|>`. Model sẽ học dự đoán "token tiếp theo là image_pad", một việc vô nghĩa vì lúc suy luận các vị trí này luôn được cho sẵn.
- Prompt giống hệt nhau ở mọi mẫu, nên model chỉ học thuộc lòng.
- Khoảng 81% token mang loss dễ, nên tín hiệu gradient của 72 token toạ độ bị pha loãng khoảng 5 lần, và loss trung bình trông thấp một cách giả tạo.
- Model cũng có thể học sinh lại văn bản prompt, làm hỏng định dạng đầu ra.

Mình không chạy thử biến thể không mask; phần trên là phân tích dựa trên số đo token.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

**Vì sao không huấn luyện được.** Khi lượng tử hoá, các `nn.Linear` của merger bị thay bằng `bnb.Linear4bit`. Trọng số lúc đó là `Params4bit` dạng uint8 đóng gói, 2 giá trị/byte, kèm `quant_state` (absmax theo block). Lúc forward, trọng số được giải nén tạm thời. Không có tensor số thực nào để nhận gradient và cập nhật, nên chỉ gắn được LoRA lên trên chứ không cập nhật trực tiếp được.

**Mình đã gặp đúng lỗi này.** Mình cấu hình `llm_int8_skip_modules=["visual", "lm_head"]` theo GUIDE:
- transformers 5.19 so khớp tên bằng `re.match`, tức là từ **đầu** tên module (hoặc so đuôi tên). Tên thật lại là `model.visual.blocks…` và `model.visual.merger…`, nên `"visual"` không khớp và **104 Linear của ViT và projector bị nén 4-bit**.
- Dấu hiệu: số tham số projector khi mở huấn luyện chỉ là 54,579,200, khoảng một nửa 109,105,152 vì trọng số đóng gói. E2 lỗi ngay bước đầu với `ValueError: Attempting to unscale FP16 gradients`.
- Mình sửa thành `["model.visual", "lm_head"]` và thêm assert sau khi nạp: 0 `Linear4bit` trong ViT, đúng 252 `Linear4bit` (LLM), projector mở đúng 109,105,152 tham số. Sau đó mình chạy lại toàn bộ E0–E3.
- Ở lần chạy lỗi, E1 với ViT 4-bit đạt F1 0.255 (CI95 [0.219, 0.295]). Với ViT fp16 là 0.279. Hai CI chồng nhau nên mình không kết luận ViT 4-bit làm giảm F1, chỉ ghi nhận.

**Vì sao projector phải ở float32:**
- Trong vòng train, mình dùng `torch.amp.GradScaler` với autocast fp16. Tham số có `requires_grad` phải là fp32 (master weights); với gradient fp16, `scaler.unscale_` báo đúng lỗi ở trên.
- fp16 chỉ có 10 bit phần định trị (nửa khoảng cách giữa hai số liền kề khoảng 4.9e-4, tính tương đối). Cập nhật nào nhỏ hơn mức đó so với trọng số sẽ bị làm tròn mất: cuối lịch cosine khi lr → 0, hoặc ở các trọng số lớn như weight của LayerNorm (≈ 1). Lưu trọng số gốc ở fp32 giữ lại được các cập nhật nhỏ này; đây là cách chuẩn của mixed precision. Phép tính forward vẫn chạy ở fp16 nhờ autocast nên vẫn nhanh.
- Lúc suy luận, projector đã train được nạp lại và chép vào model fp16. Mình in độ lệch |Δ| so với bản gốc để chắc đã nạp đúng, vì `save_pretrained` của peft chỉ lưu LoRA.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Bộ dữ liệu có **18 phiên quay**, trung bình **111.6 ảnh mỗi phiên**. Mỗi ảnh là một frame cắt từ video dashcam: các frame liền kề gần như trùng nhau (cùng đoạn đường, cùng ổ gà, cùng ánh sáng).

Mình thử chia ngẫu nhiên theo ảnh, giữ nguyên 311 ảnh test: **311/311 ảnh test (100%) có ảnh cùng phiên nằm trong train**. Chia theo phiên như `prepare_data.py` thì con số này là 0/311. Mình đã kiểm tra không phiên nào nằm ở hai split.

Khi chia ngẫu nhiên, gần như ảnh test nào cũng có "anh em" gần giống hệt trong train. Model chỉ cần nhớ cảnh (xe phía trước, mảng vỉa hè, vị trí ổ gà quen) là khớp được box, nên F1 trên test đo khả năng ghi nhớ chứ không đo khả năng tổng quát hoá sang đường mới. Trên đường mới hiệu năng sẽ thấp hơn nhiều. Mình không train trên split ngẫu nhiên nên không có con số F1 giả để so sánh; kết luận dựa trên mức trùng phiên đo được.

| Split | Phiên | Ảnh | pothole | crack | manhole | box/ảnh |
|---|---|---|---|---|---|---|
| train | 13 | 1,422 | 1,074 | 1,750 | 691 | 2.47 |
| val | 2 | 276 | 66 | 470 | 103 | 2.32 |
| test | 3 | 311 | 121 | 299 | 163 | 1.87 |

Không ảnh nào rỗng. Phân bố lớp lệch giữa các split (val chỉ có 66 pothole), nên mình chỉ dùng 40 ảnh val để gỡ lỗi và đọc F1 theo từng lớp.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

Mình dùng paired bootstrap: lấy mẫu lại cùng một tập ảnh cho cả hai setup, 2,000 lần.

| So sánh | ΔF1 | CI95 ghép cặp của ΔF1 | ΔVRAM | Δ thời gian |
|---|---|---|---|---|
| E1 → E2 (+ projector) | +0.087 | [+0.050, +0.126] | +1.70 GB | +5.1 phút |
| E2 → E3 (+ LoRA ViT) | +0.053 | [+0.010, +0.096] | +0.17 GB | +6.7 phút |
| E1 → E3 | +0.140 | [+0.095, +0.184] | +1.87 GB | +11.8 phút |

- **Mở projector rất đáng.**
  - F1 tăng 0.087 với CI ghép cặp hoàn toàn dương, FP/ảnh giảm từ 1.36 xuống 0.97.
  - Mức tăng VRAM 1.70 GB khớp với ước tính: 109M tham số fp32 (0.44 GB) + gradient fp32 (0.44 GB) + hai moment của AdamW (0.87 GB), trừ đi bản fp16 cũ (0.22 GB), được khoảng 1.5 GB.
  - Lớp crack hưởng lợi nhiều nhất (0.211 → 0.316): projector "dịch" đặc trưng thị giác sang không gian của LLM, mà kết cấu vết nứt là thứ model gốc chưa từng được dạy gọi tên.
- **LoRA trên ViT cũng có lợi nhưng ít hơn.**
  - F1 tăng 0.053 và CI ghép cặp vẫn dương, dù cận dưới chỉ +0.010. Pothole tăng nhiều nhất (0.192 → 0.284); đây là lớp nhỏ nhất, cần đặc trưng ViT tinh hơn.
  - Chi phí VRAM gần như bằng 0 (+0.17 GB, vì LoRA ViT chỉ thêm 6.3M tham số). Thời gian tăng 6.7 phút do backward phải đi qua 24 block ViT.
- **Đáng đồng tiền nhất là E3**, vì VRAM đỉnh 8.22 GB vẫn còn rộng so với 16 GB của T4. Nếu bị giới hạn thời gian, E2 cho phần lớn mức tăng. Khoảng 60% mức tăng E1 → E3 đến từ projector.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

**Độ rộng CI95 bootstrap theo ảnh** (1,000 lần):
- E0: 0.051; E1: 0.075; E2: 0.088; E3: 0.087.
- Tức khoảng ±0.04 quanh F1, gần bằng một nửa mức chênh lệch giữa hai setup liền kề.

**Hệ quả với kết luận:**
- **Nhìn hai CI riêng lẻ** không đủ: CI của E2 [0.323, 0.411] và E3 [0.377, 0.464] **chồng nhau**. Theo tiêu chí này, mình không dám nói E3 hơn E2.
- **Bootstrap ghép cặp** chặt hơn, vì cùng một ảnh khó thì khó cho cả hai setup và phần dao động chung bị triệt tiêu. ΔF1 = +0.053 với CI [+0.010, +0.096] không chứa 0, nhưng cận dưới sát 0. Kết luận "E3 tốt hơn E2" chỉ ở mức yếu.
- Các so sánh E0 → E1, E1 → E2, E1 → E3 thì vững: CI riêng lẻ tách rời và CI ghép cặp dương rõ ràng.

**Hai giới hạn khác làm CI thực tế còn rộng hơn số in ra:**
1. **Test chỉ có 3 phiên quay.** Ảnh trong cùng phiên tương quan mạnh, mà bootstrap lại coi 311 ảnh là độc lập, nên đánh giá thấp độ bất định. Thực chất cỡ mẫu hiệu dụng nhỏ hơn 311 nhiều, và kết quả có thể phụ thuộc vào 3 đoạn đường này.
2. **Mỗi setup chỉ chạy một seed**, nên CI không chứa dao động do huấn luyện. Ngoài ra mình chọn E3 làm setup gốc cho E4 theo F1 test, nên con số của E3 có thể hơi lạc quan.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

F1 từng lớp của E3:
- **pothole 0.284** (P 0.30, R 0.27): kém nhất.
- **crack 0.340** (P 0.34, R 0.34).
- **manhole 0.690** (P 0.80, R 0.61).

| Lớp | Diện tích box trung vị (train) | Box train | Box test |
|---|---|---|---|
| pothole | 0.38% khung hình | 1,074 | 121 |
| crack | 1.35% | 1,750 | 299 |
| manhole | 0.39% | 691 | 163 |

Phân tích:
- **Pothole kém nhất dù có nhiều mẫu hơn manhole.** Kích thước không giải thích hết, vì manhole cũng nhỏ (0.39%) mà đạt F1 0.69. Khác biệt nằm ở **ngữ nghĩa và hình thái**:
  - Manhole là vật thể có hình dạng rõ (tròn hoặc chữ nhật, viền kim loại) và tách biệt khỏi nền.
  - Ổ gà là mảng nhựa sẫm màu, ranh giới mờ, dễ lẫn với vệt ướt, bóng cây, mảng vá. Ổ gà lại thường xuất hiện thành cụm nhiều box sát nhau, nên IoU dễ dưới 0.5.
  - Ở `per_class_errors.png` (cột 1, `vlcsnap-2025-02-19-17h03m28s017`): 5 ổ gà nhỏ ở xa, E3 bỏ sót 5 (FN = 5) và đặt 2 box lệch.
- **Crack kém vì ranh giới nhãn mơ hồ.** Vết nứt là cấu trúc dạng đường, có thể gán nhãn thành một box lớn hoặc nhiều box nhỏ. Ở `pred_examples.png` (hai ảnh "SAI" đầu hàng dưới) và `per_class_errors.png` (cột 2): GT có 5 box crack nhỏ, còn E3 vẽ **một box lớn bao trùm cả cụm**. Box đó chứa đúng vùng hư hại nhưng IoU với từng box GT < 0.5, nên bị tính 1 FP + 5 FN.
- Đây cũng là lớp mà model gốc yếu nhất: E0 có F1 crack 0.032, model dán nhãn "crack" cho 201 vật thể nhưng gần như không cái nào khớp.
- Ở các ảnh "ĐÚNG", E3 khớp cả ổ gà nhỏ và manhole với IoU cao (mean IoU của các cặp khớp là 0.697).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

**Biến được đổi:** `data_size` (10% vs 100% ảnh train). Giữ nguyên mọi thứ khác của setup gốc E3: LoRA LLM + ViT r=16, projector fp32, lr 2e-4, 300 bước × 8 ảnh. Bản 10% dùng 142 ảnh, lấy bằng seed 42.

| | F1 | CI95 | FP/ảnh | pothole | crack | manhole |
|---|---|---|---|---|---|---|
| 10% (142 ảnh) | 0.267 | [0.235, 0.303] | 1.73 | 0.208 | 0.184 | 0.547 |
| 100% (1,422 ảnh) | 0.418 | [0.377, 0.464] | 0.97 | 0.284 | 0.340 | 0.690 |

- **Δ F1 = +0.151**, CI95 ghép cặp [+0.110, +0.194]. Hai CI riêng lẻ tách rời. Theo cả hai tiêu chí, dữ liệu nhiều hơn tốt hơn rõ ràng.
- **Vì sao 10% kém:** cùng 300 bước nhưng chỉ có 142 ảnh, tức khoảng 17 epoch.
  - Loss train về 0.0054 ở bước 300 (`loss_e4_data_0.1.png`), trong khi E3 dừng ở khoảng 0.50. Model học thuộc 142 ảnh, với 148M tham số huấn luyện.
  - Hệ quả: FP/ảnh tăng từ 0.97 lên 1.73, precision giảm từ 0.437 xuống 0.243 (model "thấy" hư hại ở khắp nơi), và crack giảm mạnh nhất (0.340 → 0.184).
- **Bản 10% với E3 vẫn ≈ E1 dùng 100% dữ liệu** (0.267 vs 0.279, CI chồng nhau). Mở projector và ViT bù lại được phần nào việc thiếu dữ liệu, nhưng không thay thế được dữ liệu.
- **Độ tin cậy:** chênh lệch lớn gấp khoảng 3–4 lần nửa độ rộng CI (≈ 0.04), và cận dưới ghép cặp là +0.110, nên kết luận về **hướng** là đáng tin. Độ lớn chính xác vẫn chịu các giới hạn ở câu 8: test chỉ có 3 phiên, một seed, và 142 ảnh được lấy ngẫu nhiên nên có thể lệch phiên. Thời gian và VRAM gần như không đổi (53.3 phút, 7.67 GB) vì số bước không đổi: giảm dữ liệu không tiết kiệm tính toán.

## Nhìn lại

**Những lỗi công cụ chấm không bắt được, mình chỉ phát hiện nhờ kiểm tra thêm:**
1. **`llm_int8_skip_modules=["visual"]` không có tác dụng** với transformers 5.19 (xem câu 5). E0/E1 vẫn chạy và vẫn đạt tiêu chí, nên `score.py` và `check_submission.py` hoàn toàn không phát hiện ViT đã bị nén 4-bit. Mình chỉ thấy khi E2 lỗi và số tham số projector bằng một nửa. Từ đó mình thêm assert ngay sau khi nạp model.
2. **peft 0.21.2 báo lỗi với torchao 0.10 có sẵn trên Colab**, chỉ khi gắn LoRA lên Linear fp16 (ViT ở E3). E1/E2 không bị vì LoRA chỉ gắn lên `Linear4bit`. Mình gỡ torchao để xử lý.
3. **trl 1.9.2 mặc định `bf16=True`, và tự ép tham số LoRA của model lượng tử về bf16.** Trên T4, với lr 5e-6 của GRPO, cập nhật bf16 bị làm tròn mất. Mình đặt `fp16=True` và đưa LoRA về fp32. Ngoài ra `remove_unused_columns=True` (mặc định) sẽ xoá cột ground truth trước khi gọi hàm reward.
4. Checkpoint gợi ý patch 14×14 cho 220 token. Processor thật dùng **patch 16** (grid 22×40 → 220 token sau khi gộp 2×2).

**Những điều mình chưa chắc:**
- Nguyên nhân cụ thể của 51 output JSON hỏng ở E0: bị cắt ở 256 token, hay viết thêm chữ thừa.
- Nếu đánh giá trên nhiều phiên quay hơn, thứ hạng E2/E3 có giữ nguyên không.

**E5 (GRPO, chưa xong):**
- Hàm reward `lab/reward.py` có 11 test, đạt 11/11.
- GRPO đã chạy 60 bước từ adapter E3: 39,321,600 tham số LoRA, 5.65 GB VRAM, 27.7 phút, dùng 1,280 ảnh train có đáp án < 120 token.
- Log: reward dao động 0.31–0.48 qua 60 bước, không có xu hướng tăng rõ. `reward_std` > 0 và `frac_reward_zero_std` = 0 ở mọi bước, nên nhóm 4 completion luôn có tín hiệu so sánh. Độ dài completion trung bình 33–77 token, không sụp về khoảng 2 token, nên chưa thấy dấu hiệu reward hacking kiểu trả `[]` hàng loạt.
- Session bị ngắt trước khi suy luận và chấm xong E5, nên mình chưa có F1 của E5 để so với SFT.
