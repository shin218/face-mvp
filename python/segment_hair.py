#!/usr/bin/env python3
"""
Hair Segmentation: GroundingDINO (HuggingFace Transformers) + SAM (Meta)

사용법:
    python segment_hair.py <image_path> [--output-dir <dir>]

출력:
    <output-dir>/hair_mask.png       흰색=hair 영역
    <output-dir>/hair_mask_inv.png   반전 마스크

의존성 설치:
    pip install -r requirements.txt
    # SAM 체크포인트는 자동 다운로드 (첫 실행 시 ~375MB)
"""

import sys
import os
import argparse
import urllib.request
import numpy as np
import cv2
from PIL import Image
import torch

# ── 설정 ─────────────────────────────────────────────────────────────────────

GROUNDING_DINO_MODEL  = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT           = "hair."          # GroundingDINO 은 마침표 필요
BOX_THRESHOLD         = 0.25
TEXT_THRESHOLD        = 0.20

SAM_MODEL_TYPE        = "vit_b"
SAM_CHECKPOINT_URL    = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_PATH   = os.path.join(os.path.dirname(__file__), "sam_vit_b_01ec64.pth")

# 얼굴 경계 침범 제거를 위한 erosion 설정
EROSION_KERNEL_SIZE   = 5    # px
EROSION_ITER_GLOBAL   = 2    # 전체 마스크 erosion 횟수
EROSION_ITER_LOWER    = 4    # 이미지 하단 40% 추가 erosion 횟수

# 노이즈 제거: 전체 픽셀 대비 이 비율 이하 blob은 제거
MIN_BLOB_RATIO        = 0.005   # 0.5%

# OpenCV Haar cascade (내장): 얼굴 bbox 보정에 사용
FACE_CASCADE_PATH     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─────────────────────────────────────────────────────────────────────────────


def download_sam_checkpoint():
    """SAM vit_b 체크포인트 자동 다운로드 (없을 경우)"""
    if os.path.exists(SAM_CHECKPOINT_PATH):
        return
    print(f"[SAM] 체크포인트가 없습니다. 다운로드 시작 ({SAM_CHECKPOINT_URL})")
    print(f"[SAM] 저장 위치: {SAM_CHECKPOINT_PATH}  (~375 MB)")
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH) or ".", exist_ok=True)

    def _progress(count, block_size, total_size):
        done = count * block_size
        pct  = min(done / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r  {done / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB  ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH, reporthook=_progress)
    print("\n[SAM] 다운로드 완료")


def load_grounding_dino(device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print(f"[GROUNDING_DINO] 모델 로드: {GROUNDING_DINO_MODEL}")
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(device)
    return processor, model


def detect_hair_boxes(image_pil, processor, model, device):
    """GroundingDINO로 hair bounding box (xyxy) 탐지"""
    inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,               # transformers 5.x: box_threshold → threshold
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],   # (H, W)
    )

    boxes  = results[0]["boxes"].cpu().numpy()   # shape [N, 4] xyxy float
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0].get("labels", ["hair"] * len(boxes))

    for label, score, box in zip(labels, scores, boxes):
        print(f"  탐지: {label}  score={score:.3f}  box={[round(v) for v in box]}")

    return boxes, scores


def detect_face_box(image_np):
    """OpenCV Haar cascade로 얼굴 bbox 탐지 (있으면 [x,y,w,h], 없으면 None)"""
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    gray    = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴 선택
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    print(f"[FACE] 얼굴 탐지: x={x} y={y} w={w} h={h}")
    return x, y, w, h


def build_hair_mask_sam(image_np, boxes, device):
    """SAM으로 hair boxes → 정밀 마스크 생성"""
    from segment_anything import SamPredictor, sam_model_registry

    download_sam_checkpoint()
    print(f"[SAM] 체크포인트 로드: {SAM_CHECKPOINT_PATH}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_np)

    H, W  = image_np.shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)

    for i, box in enumerate(boxes):
        # box: [x1, y1, x2, y2] float → int
        box_int = np.array([
            max(0, int(box[0])),
            max(0, int(box[1])),
            min(W, int(box[2])),
            min(H, int(box[3])),
        ])
        masks, iou_preds, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_int[None, :],     # (1, 4)
            multimask_output=True,    # 여러 후보 중 최고 점수 선택
        )
        best_idx = int(np.argmax(iou_preds))
        mask     = (masks[best_idx] * 255).astype(np.uint8)
        print(f"  SAM box {i+1}: iou={iou_preds[best_idx]:.3f}  coverage={mask.mean()/255*100:.1f}%")
        combined = np.maximum(combined, mask)

    return combined


def fallback_top_box_sam(image_np, device):
    """GroundingDINO 탐지 실패 시: 이미지 상단 55% 를 박스로 SAM 실행"""
    H, W = image_np.shape[:2]
    fallback_box = np.array([[0, 0, W, int(H * 0.55)]], dtype=float)
    print("[FALLBACK] hair 탐지 실패 → 상단 55% 박스로 SAM 실행")
    return build_hair_mask_sam(image_np, fallback_box, device)


def postprocess_mask(mask, image_np, face_box):
    """
    1. 작은 blob 노이즈 제거
    2. 얼굴 영역 침범 제거 (face_box 기반 + 상단 40% 이하 erosion)
    3. 마스크 shrink (erosion) 으로 경계 침범 최소화
    4. Morphological close 로 구멍 채우기
    """
    H, W = mask.shape[:2]
    result = mask.copy()

    # ── 1. 노이즈 제거 (작은 connected component 제거) ───────────────────────
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    min_area = H * W * MIN_BLOB_RATIO
    clean = np.zeros_like(result)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels_cc == i] = 255
    result = clean

    # ── 2. 얼굴 bbox 기반 마스크 제거 ───────────────────────────────────────
    if face_box is not None:
        fx, fy, fw, fh = face_box
        # 얼굴 box 를 약간 padding 하여 제거 범위 확대
        pad_x = int(fw * 0.05)
        pad_y = int(fh * 0.05)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(W, fx + fw + pad_x)
        y2 = min(H, fy + fh + pad_y)
        # 얼굴의 상단 50%는 제거하지 않음 (이마/머리카락 경계)
        # 얼굴의 하단 50%(코/입/턱)는 확실히 제거
        face_lower_y1 = fy + int(fh * 0.45)
        result[face_lower_y1:y2, x1:x2] = 0
        print(f"[POST] 얼굴 하단 제거: y={face_lower_y1}~{y2}, x={x1}~{x2}")

    # ── 3. 전체 erosion (경계 shrink) ────────────────────────────────────────
    k_global = np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), np.uint8)
    result = cv2.erode(result, k_global, iterations=EROSION_ITER_GLOBAL)

    # ── 4. 이미지 하단 40% (얼굴 근처) 추가 erosion ─────────────────────────
    boundary_y = int(H * 0.60)    # 이미지 높이 60% 아래부터 강화
    k_lower    = np.ones((7, 7), np.uint8)
    lower_part = result[boundary_y:, :]
    lower_part = cv2.erode(lower_part, k_lower, iterations=EROSION_ITER_LOWER)
    result[boundary_y:, :] = lower_part

    # ── 5. Morphological close (작은 구멍 채우기) ────────────────────────────
    k_close = np.ones((9, 9), np.uint8)
    result  = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_close, iterations=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Hair Segmentation: GroundingDINO + SAM")
    parser.add_argument("image_path",   help="입력 이미지 경로")
    parser.add_argument("--output-dir", default=".", help="출력 디렉토리 (기본: 현재 디렉토리)")
    parser.add_argument("--no-gpu",     action="store_true", help="CPU 강제 사용")
    args = parser.parse_args()

    # ── 기본 검사 ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.image_path):
        print(f"[ERROR] 파일 없음: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] 이미지: {args.image_path}")
    print(f"[INFO] 출력 디렉토리: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 이미지 로드 ───────────────────────────────────────────────────────────
    image_pil = Image.open(args.image_path).convert("RGB")
    image_np  = np.array(image_pil)
    H, W      = image_np.shape[:2]
    print(f"[INFO] 이미지 크기: {W}x{H}")

    # ── Step 1: GroundingDINO ──────────────────────────────────────────────────
    try:
        processor, gd_model = load_grounding_dino(device)
        boxes, scores       = detect_hair_boxes(image_pil, processor, gd_model, device)
    except Exception as e:
        print(f"[ERROR] GroundingDINO 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: SAM 마스크 생성 ───────────────────────────────────────────────
    try:
        if len(boxes) == 0:
            raw_mask = fallback_top_box_sam(image_np, device)
        else:
            raw_mask = build_hair_mask_sam(image_np, boxes, device)
    except Exception as e:
        print(f"[ERROR] SAM 실패: {e}", file=sys.stderr)
        sys.exit(1)

    coverage_raw = np.count_nonzero(raw_mask) / (H * W) * 100
    print(f"[SAM] raw 마스크 hair 비율: {coverage_raw:.1f}%")

    # ── Step 3: 얼굴 탐지 ────────────────────────────────────────────────────
    face_box = detect_face_box(image_np)
    if face_box is None:
        print("[FACE] 얼굴 미탐지 → 위치 기반 후처리만 진행")

    # ── Step 4: 후처리 ───────────────────────────────────────────────────────
    final_mask = postprocess_mask(raw_mask, image_np, face_box)
    coverage   = np.count_nonzero(final_mask) / (H * W) * 100
    print(f"[POST] 최종 hair 비율: {coverage:.1f}%")

    if coverage < 0.5:
        print("[WARN] hair 비율이 너무 낮습니다 (<0.5%). 마스크를 확인하세요.")

    # ── Step 5: 저장 ─────────────────────────────────────────────────────────
    hair_mask_path     = os.path.join(args.output_dir, "hair_mask.png")
    hair_mask_inv_path = os.path.join(args.output_dir, "hair_mask_inv.png")

    cv2.imwrite(hair_mask_path,     final_mask)
    cv2.imwrite(hair_mask_inv_path, 255 - final_mask)

    print(f"[DONE] hair_mask.png     → {hair_mask_path}")
    print(f"[DONE] hair_mask_inv.png → {hair_mask_inv_path}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
