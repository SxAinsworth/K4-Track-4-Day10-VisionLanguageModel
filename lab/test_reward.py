"""Test hàm reward E5. Chạy: python -m pytest lab -q"""
import json

import pytest

from lab.reward import compute_reward, iou_reward

POT = {"label": "pothole", "bbox": (100, 100, 200, 200)}
CRK = {"label": "crack", "bbox": (500, 500, 700, 600)}


def ans(*boxes):
    return json.dumps([{"label": b["label"], "bbox_2d": list(b["bbox"])} for b in boxes])


def test_perfect():
    assert compute_reward(ans(POT, CRK), [POT, CRK]) == pytest.approx(1.0)


def test_broken_json():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [100, 100', [POT]) == 0.0
    assert compute_reward("Based on the image, there is a pothole.", [POT]) == 0.0


def test_wrong_label():
    wrong = {"label": "manhole", "bbox": POT["bbox"]}
    assert compute_reward(ans(wrong), [POT]) == pytest.approx(0.2)


def test_extra_box():
    # 1 GT, dự đoán đúng box đó + 1 box bịa → 0.2 + 0.8 * 1 / 2
    assert compute_reward(ans(POT, CRK), [POT]) == pytest.approx(0.6)


def test_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_with_gt():
    # Trả [] cho ảnh có hư hại chỉ được điểm định dạng
    assert compute_reward("[]", [POT, CRK]) == pytest.approx(0.2)


def test_giant_box_not_rewarded():
    giant = {"label": "pothole", "bbox": (0, 0, 1000, 1000)}
    assert compute_reward(ans(giant), [POT]) == pytest.approx(0.2)


def test_partial_iou_and_threshold():
    shifted = {"label": "pothole", "bbox": (110, 100, 210, 200)}      # IoU = 90/110
    assert compute_reward(ans(shifted), [POT]) == pytest.approx(0.2 + 0.8 * 90 / 110)
    far = {"label": "pothole", "bbox": (160, 100, 260, 200)}          # IoU = 40/160 < 0.5
    assert compute_reward(ans(far), [POT]) == pytest.approx(0.2)


def test_pixel_scale_not_rewarded():
    # Cùng box nhưng ghi theo pixel 640×360 → lệch thang, IoU < 0.5
    pix = {"label": "pothole", "bbox": (64, 36, 128, 72)}
    assert compute_reward(ans(pix), [POT]) == pytest.approx(0.2)


def test_fenced_output_parsed_like_scorer():
    assert compute_reward("```json\n" + ans(POT) + "\n```", [POT]) == pytest.approx(1.0)


def test_trl_wrapper_conversational_completions():
    gt = json.dumps([{"label": "pothole", "bbox": list(POT["bbox"])}])
    completions = [[{"role": "assistant", "content": ans(POT)}], [{"role": "assistant", "content": "oops"}]]
    assert iou_reward(prompts=None, completions=completions, gt=[gt, gt]) == pytest.approx([1.0, 0.0])
    assert iou_reward(prompts=None, completions=["[]"], gt=["[]"]) == [1.0]
