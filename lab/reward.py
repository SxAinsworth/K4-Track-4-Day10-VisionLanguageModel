"""Reward IoU kiểm chứng được cho E5 (GRPO). Python thuần, test được không cần GPU."""
import json

from scripts.score import IOU_THR, iou_xyxy, parse_output


def compute_reward(completion, gt_boxes):
    """completion: chuỗi model sinh ra; gt_boxes: list[{"label", "bbox": (x1, y1, x2, y2)}] thang 0–1000.

    0.0 nếu JSON hỏng; 1.0 nếu cả GT và dự đoán đều rỗng; còn lại
    0.2 + 0.8 * (tổng IoU các cặp khớp) / max(n_gt, n_pred, 1).
    Ghép đôi tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU ≥ 0.5 (giống trọng tài).
    Box thừa làm tăng mẫu số → bị phạt; box khổng lồ có IoU < 0.5 với box nhỏ → không được điểm.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total_iou += v
    return 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(pred), 1)


def _completion_text(c):
    """trl truyền completion dạng chuỗi, hoặc dạng hội thoại [{"role": "assistant", "content": ...}]."""
    if isinstance(c, list):
        c = c[-1]["content"]
    if isinstance(c, list):
        c = "".join(part.get("text", "") for part in c if isinstance(part, dict))
    return c


def iou_reward(prompts, completions, gt, **kwargs):
    """Hàm reward theo chữ ký của trl.GRPOTrainer; cột `gt` của dataset là JSON list[{"label","bbox"}]."""
    out = []
    for c, g in zip(completions, gt):
        boxes = [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(g)]
        out.append(compute_reward(_completion_text(c), boxes))
    return out
