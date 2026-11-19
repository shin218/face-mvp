"""
compare_landmarks.py

MediaPipe Face Mesh vs Face++ 106p Landmark 비교 스크립트.

사용법:
    python python/compare_landmarks.py <이미지_경로_또는_폴더>

출력:
    output/compare/<stem>_mediapipe.png   – MediaPipe 특징점 시각화
    output/compare/<stem>_facepp.png      – Face++ 특징점 시각화
    output/compare/<stem>_compare.png    – 나란히 붙인 비교 이미지
    output/compare/metrics.csv           – 지표 비교표
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

# ── 환경변수 ─────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

FACEPP_API_KEY    = os.getenv("FACEPP_API_KEY", "")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET", "")
FACEPP_DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# ── 출력 폴더 ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "compare"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MediaPipe 초기화 ──────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 특징점 추출
# ═══════════════════════════════════════════════════════════════════════════════

def _imread(image_path: Path) -> Optional[np.ndarray]:
    """한글/유니코드 경로를 처리하는 cv2.imread 대체."""
    buf = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def extract_mediapipe(image_path: Path) -> Optional[dict]:
    """
    MediaPipe Face Landmarker(Face Mesh)로 특징점 추출.
    Returns: { "landmarks": [(x_px, y_px), ...] 468개, "w": int, "h": int }
             정면 얼굴이 아니거나 미검출이면 None, reason 문자열 반환
    """
    img_bgr = _imread(image_path)
    if img_bgr is None:
        return None, "이미지를 읽을 수 없습니다"

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return None, "MediaPipe: 얼굴 미검출"

    lms = result.multi_face_landmarks[0].landmark
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    # 정면 여부: nose_tip(1), left_eye(33), right_eye(263) 기준 좌우 대칭 체크
    nose  = pts[1]
    leye  = pts[33]
    reye  = pts[263]
    eye_cx = (leye[0] + reye[0]) / 2
    off   = abs(nose[0] - eye_cx) / max(abs(reye[0] - leye[0]), 1)
    if off > 0.3:
        return None, f"MediaPipe: 정면 아님 (비대칭 오프셋 {off:.2f})"

    return {"landmarks": pts, "w": w, "h": h}, None


def extract_facepp(image_path: Path) -> Optional[dict]:
    """
    Face++ Detect API (return_landmark=1) 단일 호출로 83p 랜드마크 획득.
    별도 Landmark API(/facepp/v3/face/landmark)는 플랜에 따라 비활성화될 수 있어,
    Detect 내장 landmark(83p)를 사용한다.
    Returns: { "landmark": {name: {x,y}, ...}, "face_rect": {w,h,...} }
    """
    if not FACEPP_API_KEY or not FACEPP_API_SECRET:
        return None, "Face++ 키 없음 (.env에 FACEPP_API_KEY / FACEPP_API_SECRET 필요)"

    image_bytes = image_path.read_bytes()
    if not image_bytes:
        return None, f"Face++: 이미지 파일이 비어 있음 ({image_path.name})"

    try:
        resp = requests.post(
            FACEPP_DETECT_URL,
            data={
                "api_key":           FACEPP_API_KEY,
                "api_secret":        FACEPP_API_SECRET,
                "return_attributes": "headpose",
                "return_landmark":   "1",
            },
            files={"image_file": (image_path.name, image_bytes)},
            timeout=30,
        )
        resp.raise_for_status()
        det = resp.json()
    except Exception as e:
        return None, f"Face++ Detect 오류: {e}"

    if det.get("error_message"):
        return None, f"Face++ API 오류: {det['error_message']}"

    faces = det.get("faces", [])
    if not faces:
        return None, "Face++: 얼굴 미검출"

    face = faces[0]

    # 정면 체크 (headpose)
    attrs   = face.get("attributes", {})
    headpose = attrs.get("headpose", {})
    yaw   = abs(headpose.get("yaw_angle", 0))
    pitch = abs(headpose.get("pitch_angle", 0))
    if yaw > 30 or pitch > 30:
        return None, f"Face++: 정면 아님 (yaw={yaw:.1f}, pitch={pitch:.1f})"

    landmark = face.get("landmark", {})
    if not landmark:
        return None, "Face++: Landmark 비어 있음 (return_landmark=1 응답 없음)"

    rect = face.get("face_rectangle", {})
    return {
        "landmark": landmark,
        "face_rect": rect,
    }, None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 지표 계산
# ═══════════════════════════════════════════════════════════════════════════════

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ── MediaPipe 랜드마크 인덱스 (Face Mesh 468pt) ──────────────────────────────
# 참고: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
MP_IDX = {
    # 얼굴 외곽 (face oval): 위에서 아래
    "face_top":     10,    # 이마 중앙 (헤어라인 아님 — 추정치)
    "chin":        152,    # 턱 끝
    "left_cheek":  234,    # 얼굴 왼쪽 가장 넓은 점
    "right_cheek": 454,    # 얼굴 오른쪽 가장 넓은 점
    # 눈
    "l_eye_inner": 133,    # 왼눈 안쪽
    "l_eye_outer":  33,    # 왼눈 바깥쪽
    "r_eye_inner": 362,    # 오른눈 안쪽
    "r_eye_outer": 263,    # 오른눈 바깥쪽
    "l_eye_center": 468,   # (refine) 왼눈 홍채 중심
    "r_eye_center": 473,   # (refine) 오른눈 홍채 중심
    # 눈썹
    "l_brow_upper": 105,   # 왼 눈썹 상단 중앙
    "r_brow_upper": 334,   # 오른 눈썹 상단 중앙
    # 코
    "nose_tip":      1,
    "nose_l":       218,   # 코 왼쪽 날개
    "nose_r":       438,   # 코 오른쪽 날개
    # 입
    "mouth_l":      61,    # 입꼬리 왼쪽
    "mouth_r":     291,    # 입꼬리 오른쪽
    "upper_lip_top": 0,    # 윗입술 상단 중앙
    "upper_lip_bot": 13,   # 윗입술 하단 (입 열렸을 때 경계)
    "lower_lip_top": 14,   # 아랫입술 상단
    "lower_lip_bot": 17,   # 아랫입술 하단 중앙
    # 턱 윤곽
    "jaw_l":        172,   # 턱 왼쪽
    "jaw_r":        397,   # 턱 오른쪽
    # 중안부 경계
    "nose_bridge":  168,   # 미간 (눈썹 사이 코 뿌리)
}


def metrics_from_mediapipe(data: dict) -> dict:
    pts = data["landmarks"]
    w, h = data["w"], data["h"]

    def g(name):
        return pts[MP_IDX[name]]

    face_top    = g("face_top")
    chin        = g("chin")
    left_cheek  = g("left_cheek")
    right_cheek = g("right_cheek")

    face_w = _dist(left_cheek, right_cheek)
    face_h = _dist(face_top, chin)

    # 1. 얼굴 가로/세로 비율
    face_wh_ratio = face_w / face_h if face_h > 0 else None

    # 2. 얼굴 삼등분 (이마/중안부/하안부)
    # 이마: face_top → 눈썹 상단(근사 헤어라인 대용 — 추정치)
    brow_mid_y  = (g("l_brow_upper")[1] + g("r_brow_upper")[1]) / 2
    nose_tip_y  = g("nose_tip")[1]
    chin_y      = chin[1]
    face_top_y  = face_top[1]

    forehead_len = abs(brow_mid_y - face_top_y)   # 추정치
    midface_len  = abs(nose_tip_y - brow_mid_y)
    lower_len    = abs(chin_y - nose_tip_y)
    total_thirds = forehead_len + midface_len + lower_len

    thirds = {
        "forehead_ratio": forehead_len / total_thirds if total_thirds > 0 else None,  # 추정치
        "midface_ratio":  midface_len  / total_thirds if total_thirds > 0 else None,
        "lower_ratio":    lower_len    / total_thirds if total_thirds > 0 else None,
    }

    # 3. 눈 사이 거리 / 눈 너비 (오등분)
    l_eye_inner = g("l_eye_inner")
    r_eye_inner = g("r_eye_inner")
    l_eye_outer = g("l_eye_outer")
    r_eye_outer = g("r_eye_outer")

    eye_gap      = _dist(l_eye_inner, r_eye_inner)   # 미간
    l_eye_width  = _dist(l_eye_outer, l_eye_inner)
    r_eye_width  = _dist(r_eye_outer, r_eye_inner)
    avg_eye_w    = (l_eye_width + r_eye_width) / 2
    eye_gap_ratio = eye_gap / avg_eye_w if avg_eye_w > 0 else None  # 오등분 기준 ≈1이 이상적

    # 4. 코 너비 / 눈 사이 거리
    nose_l = g("nose_l")
    nose_r = g("nose_r")
    nose_w = _dist(nose_l, nose_r)
    nose_eye_ratio = nose_w / eye_gap if eye_gap > 0 else None

    # 5. 입 너비 / 코 너비
    mouth_l = g("mouth_l")
    mouth_r = g("mouth_r")
    mouth_w = _dist(mouth_l, mouth_r)
    mouth_nose_ratio = mouth_w / nose_w if nose_w > 0 else None

    # 6. 윗입술 / 아랫입술 두께 비율
    upper_lip_top = g("upper_lip_top")
    upper_lip_bot = g("upper_lip_bot")
    lower_lip_top = g("lower_lip_top")
    lower_lip_bot = g("lower_lip_bot")
    upper_thick = abs(upper_lip_bot[1] - upper_lip_top[1])
    lower_thick = abs(lower_lip_bot[1] - lower_lip_top[1])
    lip_ratio = upper_thick / lower_thick if lower_thick > 0 else None

    # 7. 턱 각도 (좌우 평균)
    jaw_l = g("jaw_l")
    jaw_r = g("jaw_r")
    # 각도: chin–jaw_l, chin–jaw_r 벡터 사이 각도
    def jaw_angle(jaw_side):
        v1 = (jaw_side[0] - chin[0], jaw_side[1] - chin[1])
        v2 = (0, -1)  # 위쪽 수직 방향
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag = math.hypot(*v1)
        if mag == 0:
            return None
        cos_a = max(-1, min(1, dot / mag))
        return math.degrees(math.acos(cos_a))

    jaw_l_angle = jaw_angle(jaw_l)
    jaw_r_angle = jaw_angle(jaw_r)
    jaw_avg = (jaw_l_angle + jaw_r_angle) / 2 if jaw_l_angle and jaw_r_angle else None

    # 8. 좌우 대칭도: 주요 쌍 점들의 세로 중심선 대비 대칭 오차 평균
    cx = (left_cheek[0] + right_cheek[0]) / 2
    sym_pairs = [
        (g("l_eye_outer"), g("r_eye_outer")),
        (g("l_eye_inner"), g("r_eye_inner")),
        (g("nose_l"),      g("nose_r")),
        (g("mouth_l"),     g("mouth_r")),
        (g("jaw_l"),       g("jaw_r")),
    ]
    sym_errors = []
    for lp, rp in sym_pairs:
        ideal_r_x = 2 * cx - lp[0]
        err = abs(ideal_r_x - rp[0]) / face_w if face_w > 0 else 0
        sym_errors.append(err)
    symmetry = 1.0 - (sum(sym_errors) / len(sym_errors)) if sym_errors else None

    return {
        "face_wh_ratio":        face_wh_ratio,
        "forehead_ratio":       thirds["forehead_ratio"],   # 추정치
        "midface_ratio":        thirds["midface_ratio"],
        "lower_ratio":          thirds["lower_ratio"],
        "eye_gap_ratio":        eye_gap_ratio,
        "nose_eye_ratio":       nose_eye_ratio,
        "mouth_nose_ratio":     mouth_nose_ratio,
        "lip_ratio":            lip_ratio,
        "jaw_angle_deg":        jaw_avg,
        "symmetry":             symmetry,
        # 진단용 점 반환 (draw_diagnostic_mediapipe에서 사용)
        "_diag": {
            "face_top":     g("face_top"),
            "chin":         chin,
            "left_cheek":   left_cheek,
            "right_cheek":  right_cheek,
            "l_brow_upper": g("l_brow_upper"),
            "r_brow_upper": g("r_brow_upper"),
            "l_eye_inner":  l_eye_inner,
            "r_eye_inner":  r_eye_inner,
            "l_eye_outer":  l_eye_outer,
            "r_eye_outer":  r_eye_outer,
            "nose_tip":     g("nose_tip"),
            "nose_l":       nose_l,
            "nose_r":       nose_r,
            "mouth_l":      mouth_l,
            "mouth_r":      mouth_r,
            "upper_lip_top": upper_lip_top,
            "upper_lip_bot": upper_lip_bot,
            "lower_lip_top": lower_lip_top,
            "lower_lip_bot": lower_lip_bot,
            "jaw_l":        jaw_l,
            "jaw_r":        jaw_r,
        },
    }


def metrics_from_facepp(data: dict) -> dict:
    """
    Face++ 83p (Detect API, return_landmark=1) 기준 지표 계산.

    MediaPipe와 해부학적으로 동일한 위치의 점을 쓴다:

    지표                 | Face++ 점
    ---------------------|--------------------------------------------
    얼굴 너비(광대)       | contour_left5  ↔  contour_right5
    얼굴 높이             | eyebrow_upper_avg  ↔  contour_chin
    이마 [추정치]         | face_rect.top → eyebrow_upper_avg
                         |   (rect.top ≈ brow일 때 midface×0.9 추정)
    중안부                | eyebrow_upper_avg → nose_tip
    하안부                | nose_tip → contour_chin
    미간(눈 사이 거리)    | left_eye_right_corner ↔ right_eye_left_corner
                         |   ← 안쪽 눈구석 (MP: l_eye_inner/r_eye_inner 동일)
    눈 너비               | left_eye_left_corner  ↔ left_eye_right_corner
    콧볼 너비             | nose_left ↔ nose_right
    입 너비               | mouth_left_corner ↔ mouth_right_corner
    윗입술                | mouth_upper_lip_top → mouth_upper_lip_bottom
    아랫입술              | mouth_lower_lip_top → mouth_lower_lip_bottom
    턱 각도               | contour_left9/right9 → contour_chin 벡터
    대칭도 기준 중심선    | (contour_left5.x + contour_right5.x) / 2
    """
    lm   = data["landmark"]
    rect = data["face_rect"]

    def p(name):
        pt = lm.get(name)
        if pt is None:
            return None
        return (pt["x"], pt["y"])

    # ── 1. 얼굴 가로/세로 비율 ─────────────────────────────────────────────────
    # bounding box 대신 랜드마크 기반 치수 사용
    l_cheek5 = p("contour_left5")
    r_cheek5 = p("contour_right5")
    face_w_lm = _dist(l_cheek5, r_cheek5) if (l_cheek5 and r_cheek5) else None

    l_brow = p("left_eyebrow_upper_middle")
    r_brow = p("right_eyebrow_upper_middle")
    chin_pt = p("contour_chin")
    brow_avg = _mid(l_brow, r_brow) if (l_brow and r_brow) else None
    face_h_lm = _dist(brow_avg, chin_pt) if (brow_avg and chin_pt) else None

    face_wh_ratio = face_w_lm / face_h_lm if (face_w_lm and face_h_lm) else None

    # 대칭도 계산에 쓸 face_w
    face_w_for_sym = face_w_lm if face_w_lm else rect.get("width", 1)

    # ── 2. 얼굴 삼등분 ──────────────────────────────────────────────────────────
    nose_tip = p("nose_tip")

    if l_brow and r_brow and nose_tip and chin_pt:
        brow_y    = (l_brow[1] + r_brow[1]) / 2
        nose_y    = nose_tip[1]
        chin_y    = chin_pt[1]

        # 이마: face_rect.top 이 brow 위에 충분히 있으면 사용, 아니면 추정
        face_rect_top = rect.get("top", brow_y)
        above_brow = brow_y - face_rect_top  # 양수 = rect.top 이 brow 위에 있음

        if above_brow >= 10:
            forehead_top_y = face_rect_top
        else:
            # face_rect.top ≈ brow 수준 → 중안부 길이의 0.9배로 이마 추정 [추정치]
            midface_tmp = abs(nose_y - brow_y)
            forehead_top_y = brow_y - midface_tmp * 0.9

        forehead_len = abs(brow_y - forehead_top_y)   # [추정치]
        midface_len  = abs(nose_y - brow_y)
        lower_len    = abs(chin_y - nose_y)
        total_thirds = forehead_len + midface_len + lower_len

        forehead_r = forehead_len / total_thirds if total_thirds > 0 else None
        midface_r  = midface_len  / total_thirds if total_thirds > 0 else None
        lower_r    = lower_len    / total_thirds if total_thirds > 0 else None
    else:
        forehead_r = midface_r = lower_r = None

    # ── 3. 눈 사이 거리 / 눈 너비 (오등분) ────────────────────────────────────
    # 미간: 안쪽 눈구석끼리의 거리 (MediaPipe l_eye_inner/r_eye_inner와 동일 위치)
    #   Face++ left_eye_right_corner  = 왼눈의 오른쪽 끝 = 안쪽 눈구석
    #   Face++ right_eye_left_corner  = 오른눈의 왼쪽 끝 = 안쪽 눈구석
    l_eye_inner_fp = p("left_eye_right_corner")   # 왼눈 안쪽 (미간 쪽)
    r_eye_inner_fp = p("right_eye_left_corner")   # 오른눈 안쪽 (미간 쪽)
    l_eye_outer_fp = p("left_eye_left_corner")    # 왼눈 바깥쪽 (눈꼬리)
    r_eye_outer_fp = p("right_eye_right_corner")  # 오른눈 바깥쪽 (눈꼬리)

    eye_gap   = _dist(l_eye_inner_fp, r_eye_inner_fp) if (l_eye_inner_fp and r_eye_inner_fp) else None
    l_eye_w   = _dist(l_eye_outer_fp, l_eye_inner_fp) if (l_eye_outer_fp and l_eye_inner_fp) else None
    r_eye_w   = _dist(r_eye_inner_fp, r_eye_outer_fp) if (r_eye_inner_fp and r_eye_outer_fp) else None
    avg_eye_w = (l_eye_w + r_eye_w) / 2 if (l_eye_w and r_eye_w) else (l_eye_w or r_eye_w)
    eye_gap_ratio = eye_gap / avg_eye_w if (eye_gap and avg_eye_w) else None

    # ── 4. 코 너비 / 눈 사이 거리 ─────────────────────────────────────────────
    # nose_left / nose_right: 콧볼 가장 넓은 점 (alar base)
    nose_l = p("nose_left")
    nose_r = p("nose_right")
    nose_w = _dist(nose_l, nose_r) if (nose_l and nose_r) else None
    nose_eye_ratio = (nose_w / eye_gap) if (nose_w and eye_gap) else None

    # ── 5. 입 너비 / 코 너비 ──────────────────────────────────────────────────
    mouth_l = p("mouth_left_corner")
    mouth_r = p("mouth_right_corner")
    mouth_w = _dist(mouth_l, mouth_r) if (mouth_l and mouth_r) else None
    mouth_nose_ratio = (mouth_w / nose_w) if (mouth_w and nose_w) else None

    # ── 6. 윗입술 / 아랫입술 두께 비율 ────────────────────────────────────────
    # FacePlusService 동일 정의 (mouth_upper_lip_top→bottom, mouth_lower_lip_top→bottom)
    upper_lip_top_fp = p("mouth_upper_lip_top")
    upper_lip_bot_fp = p("mouth_upper_lip_bottom")
    lower_lip_top_fp = p("mouth_lower_lip_top")
    lower_lip_bot_fp = p("mouth_lower_lip_bottom")

    upper_thick = abs(upper_lip_bot_fp[1] - upper_lip_top_fp[1]) if (upper_lip_top_fp and upper_lip_bot_fp) else None
    lower_thick = abs(lower_lip_bot_fp[1] - lower_lip_top_fp[1]) if (lower_lip_top_fp and lower_lip_bot_fp) else None
    lip_ratio = upper_thick / lower_thick if (upper_thick and lower_thick and lower_thick > 0) else None

    # ── 7. 턱 각도 ─────────────────────────────────────────────────────────────
    contour_chin = chin_pt
    jaw_l_pt = p("contour_left9") or p("contour_left8")
    jaw_r_pt = p("contour_right9") or p("contour_right8")

    def jaw_angle(jaw_side, chin_p):
        if jaw_side is None or chin_p is None:
            return None
        v1 = (jaw_side[0] - chin_p[0], jaw_side[1] - chin_p[1])
        v2 = (0, -1)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag = math.hypot(*v1)
        if mag == 0:
            return None
        cos_a = max(-1, min(1, dot / mag))
        return math.degrees(math.acos(cos_a))

    jaw_l_ang = jaw_angle(jaw_l_pt, contour_chin)
    jaw_r_ang = jaw_angle(jaw_r_pt, contour_chin)
    jaw_avg   = (jaw_l_ang + jaw_r_ang) / 2 if (jaw_l_ang and jaw_r_ang) else None

    # ── 8. 좌우 대칭도 ─────────────────────────────────────────────────────────
    # 중심선: contour_left5 ↔ contour_right5 중간점의 x
    cx = (l_cheek5[0] + r_cheek5[0]) / 2 if (l_cheek5 and r_cheek5) else (rect.get("left", 0) + face_w_for_sym / 2)

    sym_pairs = []
    if l_eye_outer_fp and r_eye_outer_fp: sym_pairs.append((l_eye_outer_fp, r_eye_outer_fp))
    if l_eye_inner_fp and r_eye_inner_fp: sym_pairs.append((l_eye_inner_fp, r_eye_inner_fp))
    if nose_l and nose_r:                 sym_pairs.append((nose_l, nose_r))
    if mouth_l and mouth_r:               sym_pairs.append((mouth_l, mouth_r))
    if jaw_l_pt and jaw_r_pt:             sym_pairs.append((jaw_l_pt, jaw_r_pt))

    if sym_pairs and face_w_for_sym > 0:
        errs = [abs((2*cx - lp[0]) - rp[0]) / face_w_for_sym for lp, rp in sym_pairs]
        symmetry = 1.0 - sum(errs) / len(errs)
    else:
        symmetry = None

    return {
        "face_wh_ratio":    face_wh_ratio,
        "forehead_ratio":   forehead_r,    # 추정치
        "midface_ratio":    midface_r,
        "lower_ratio":      lower_r,
        "eye_gap_ratio":    eye_gap_ratio,
        "nose_eye_ratio":   nose_eye_ratio,
        "mouth_nose_ratio": mouth_nose_ratio,
        "lip_ratio":        lip_ratio,
        "jaw_angle_deg":    jaw_avg,
        "symmetry":         symmetry,
        # 진단용 점 반환 (draw_diagnostic_facepp에서 사용)
        "_diag": {
            "l_cheek5": l_cheek5, "r_cheek5": r_cheek5,
            "brow_avg": brow_avg, "chin_pt": chin_pt,
            "l_eye_inner": l_eye_inner_fp, "r_eye_inner": r_eye_inner_fp,
            "l_eye_outer": l_eye_outer_fp, "r_eye_outer": r_eye_outer_fp,
            "nose_l": nose_l, "nose_r": nose_r, "nose_tip": nose_tip,
            "mouth_l": mouth_l, "mouth_r": mouth_r,
            "upper_lip_top": upper_lip_top_fp, "upper_lip_bot": upper_lip_bot_fp,
            "lower_lip_top": lower_lip_top_fp, "lower_lip_bot": lower_lip_bot_fp,
            "jaw_l": jaw_l_pt, "jaw_r": jaw_r_pt,
            "l_brow": l_brow, "r_brow": r_brow,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 시각화
# ═══════════════════════════════════════════════════════════════════════════════

# MediaPipe Face Mesh 연결 세트 (눈/눈썹/코/입/얼굴 외곽)
MP_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

def draw_mediapipe(image_path: Path, data: dict) -> np.ndarray:
    img = _imread(image_path)
    h, w = img.shape[:2]
    pts = data["landmarks"]

    # 테셀레이션 (삼각형 메쉬) — 연한 선
    for conn in mp_face_mesh.FACEMESH_TESSELATION:
        p1 = pts[conn[0]]
        p2 = pts[conn[1]]
        cv2.line(img, p1, p2, (80, 80, 80), 1, cv2.LINE_AA)

    # 윤곽선 (굵게)
    for conn_set, color in [
        (mp_face_mesh.FACEMESH_FACE_OVAL, (0, 220, 0)),
        (mp_face_mesh.FACEMESH_LEFT_EYE,  (255, 180, 0)),
        (mp_face_mesh.FACEMESH_RIGHT_EYE, (255, 180, 0)),
        (mp_face_mesh.FACEMESH_LIPS,      (0, 100, 255)),
        (mp_face_mesh.FACEMESH_LEFT_EYEBROW,  (0, 200, 200)),
        (mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (0, 200, 200)),
    ]:
        for conn in conn_set:
            cv2.line(img, pts[conn[0]], pts[conn[1]], color, 2, cv2.LINE_AA)

    # 주요 점 표시
    for name, idx in MP_IDX.items():
        if idx < len(pts):
            cv2.circle(img, pts[idx], 4, (0, 0, 255), -1)
            cv2.putText(img, name, (pts[idx][0]+4, pts[idx][1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "MediaPipe Face Mesh", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    return img


def draw_facepp(image_path: Path, data: dict) -> np.ndarray:
    img = _imread(image_path)
    lm  = data["landmark"]

    # 모든 랜드마크 점 표시
    for name, pt in lm.items():
        x, y = int(pt["x"]), int(pt["y"])
        cv2.circle(img, (x, y), 3, (0, 180, 255), -1)
        cv2.putText(img, name, (x+3, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 200), 1, cv2.LINE_AA)

    # 얼굴 윤곽 연결 (contour_left1~9, contour_right1~9 + chin)
    contour_pts = []
    for i in range(1, 10):
        pt = lm.get(f"contour_left{i}")
        if pt:
            contour_pts.append((int(pt["x"]), int(pt["y"])))
    chin_pt = lm.get("contour_chin")
    if chin_pt:
        contour_pts.append((int(chin_pt["x"]), int(chin_pt["y"])))
    for i in range(9, 0, -1):
        pt = lm.get(f"contour_right{i}")
        if pt:
            contour_pts.append((int(pt["x"]), int(pt["y"])))
    if len(contour_pts) > 1:
        for i in range(len(contour_pts) - 1):
            cv2.line(img, contour_pts[i], contour_pts[i+1], (0, 220, 0), 2, cv2.LINE_AA)

    # 눈 윤곽
    for side in ["left", "right"]:
        eye_pts = []
        for suffix in ["left_corner", "upper_left_quarter", "upper_right_quarter", "right_corner",
                       "lower_right_quarter", "lower_left_quarter"]:
            pt = lm.get(f"{side}_eye_{suffix}")
            if pt:
                eye_pts.append((int(pt["x"]), int(pt["y"])))
        if len(eye_pts) > 1:
            for i in range(len(eye_pts)):
                cv2.line(img, eye_pts[i], eye_pts[(i+1) % len(eye_pts)], (255, 180, 0), 2)

    cv2.putText(img, "Face++ 83p (Detect API)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
    return img


# ── 진단 시각화 색상 팔레트 ────────────────────────────────────────────────────
# 각 측정 그룹을 고정 색으로 표시해 어떤 점이 어떤 지표에 쓰이는지 한눈에 보임
_C = {
    "face":    (0,   220,   0),   # 초록  – 얼굴 너비/높이
    "brow":    (0,   200, 200),   # 청록  – 눈썹(이마/중안부 경계)
    "eye":     (255, 180,   0),   # 노랑  – 눈 (미간·눈 너비)
    "nose":    (200,  60, 255),   # 보라  – 코
    "mouth":   (0,  100, 255),   # 파랑  – 입 너비
    "lip":     (0,   60, 180),   # 진파랑 – 입술 두께
    "jaw":     (255, 100,   0),   # 주황  – 턱
}

def _pt(xy) -> tuple:
    return (int(xy[0]), int(xy[1]))

def _dot(img, pt, color, r=7):
    if pt is None:
        return
    cv2.circle(img, _pt(pt), r, color, -1)
    cv2.circle(img, _pt(pt), r + 1, (0, 0, 0), 1)

def _line(img, a, b, color, t=2):
    if a is None or b is None:
        return
    cv2.line(img, _pt(a), _pt(b), color, t, cv2.LINE_AA)

def _label(img, pt, text, color):
    if pt is None:
        return
    x, y = _pt(pt)
    cv2.putText(img, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color,   1, cv2.LINE_AA)


def draw_diagnostic_mediapipe(image_path: Path, metrics: dict) -> np.ndarray:
    """
    MediaPipe: 각 지표에 실제로 쓰인 점과 선을 색상으로 구분해 표시.
    """
    img = _imread(image_path)
    d   = metrics.get("_diag", {})

    # 얼굴 너비/높이 (초록)
    _dot(img, d.get("left_cheek"),  _C["face"])
    _dot(img, d.get("right_cheek"), _C["face"])
    _dot(img, d.get("face_top"),    _C["face"])
    _dot(img, d.get("chin"),        _C["face"])
    _line(img, d.get("left_cheek"), d.get("right_cheek"), _C["face"])
    _line(img, d.get("face_top"),   d.get("chin"),        _C["face"])
    _label(img, d.get("left_cheek"),  "face_w", _C["face"])
    _label(img, d.get("face_top"),    "face_h", _C["face"])

    # 눈썹 → 삼등분 경계 (청록)
    _dot(img, d.get("l_brow_upper"), _C["brow"])
    _dot(img, d.get("r_brow_upper"), _C["brow"])
    _line(img, d.get("l_brow_upper"), d.get("r_brow_upper"), _C["brow"])
    _label(img, d.get("l_brow_upper"), "brow(이마/중안부 경계)", _C["brow"])

    # 눈 (노랑)
    _dot(img, d.get("l_eye_inner"), _C["eye"])
    _dot(img, d.get("r_eye_inner"), _C["eye"])
    _dot(img, d.get("l_eye_outer"), _C["eye"])
    _dot(img, d.get("r_eye_outer"), _C["eye"])
    _line(img, d.get("l_eye_inner"), d.get("r_eye_inner"), _C["eye"], 3)   # 미간
    _line(img, d.get("l_eye_outer"), d.get("l_eye_inner"), _C["eye"])      # 왼눈 너비
    _line(img, d.get("r_eye_inner"), d.get("r_eye_outer"), _C["eye"])      # 오른눈 너비
    _label(img, d.get("l_eye_inner"), "미간", _C["eye"])
    _label(img, d.get("l_eye_outer"), "눈너비L", _C["eye"])

    # 코 (보라)
    _dot(img, d.get("nose_l"),   _C["nose"])
    _dot(img, d.get("nose_r"),   _C["nose"])
    _dot(img, d.get("nose_tip"), _C["nose"])
    _line(img, d.get("nose_l"), d.get("nose_r"), _C["nose"])
    _label(img, d.get("nose_l"), "콧볼", _C["nose"])

    # 입 너비 (파랑)
    _dot(img, d.get("mouth_l"), _C["mouth"])
    _dot(img, d.get("mouth_r"), _C["mouth"])
    _line(img, d.get("mouth_l"), d.get("mouth_r"), _C["mouth"])
    _label(img, d.get("mouth_l"), "입너비", _C["mouth"])

    # 입술 두께 (진파랑)
    _dot(img, d.get("upper_lip_top"), _C["lip"])
    _dot(img, d.get("upper_lip_bot"), _C["lip"])
    _dot(img, d.get("lower_lip_top"), _C["lip"])
    _dot(img, d.get("lower_lip_bot"), _C["lip"])
    _line(img, d.get("upper_lip_top"), d.get("upper_lip_bot"), _C["lip"])
    _line(img, d.get("lower_lip_top"), d.get("lower_lip_bot"), _C["lip"])
    _label(img, d.get("upper_lip_top"), "윗입술", _C["lip"])
    _label(img, d.get("lower_lip_top"), "아랫입술", _C["lip"])

    # 턱 (주황)
    _dot(img, d.get("jaw_l"), _C["jaw"])
    _dot(img, d.get("jaw_r"), _C["jaw"])
    _line(img, d.get("jaw_l"), d.get("chin"), _C["jaw"])
    _line(img, d.get("jaw_r"), d.get("chin"), _C["jaw"])
    _label(img, d.get("jaw_l"), "턱각도L", _C["jaw"])

    # 범례
    legend = [
        ("초록: 얼굴 너비/높이",     _C["face"]),
        ("청록: 눈썹(삼등분 경계)",  _C["brow"]),
        ("노랑: 눈 미간·너비",       _C["eye"]),
        ("보라: 코",                 _C["nose"]),
        ("파랑: 입 너비",            _C["mouth"]),
        ("진파랑: 입술 두께",        _C["lip"]),
        ("주황: 턱 각도",            _C["jaw"]),
    ]
    for i, (text, col) in enumerate(legend):
        y = 50 + i * 20
        cv2.rectangle(img, (8, y - 12), (20, y), col, -1)
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 2)
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0),       1)

    cv2.putText(img, "MediaPipe — 측정 점 진단", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    return img


def draw_diagnostic_facepp(image_path: Path, fp_data: dict, metrics: dict) -> np.ndarray:
    """
    Face++ 83p: 각 지표에 실제로 쓰인 점과 선을 색상으로 구분해 표시.
    """
    img = _imread(image_path)
    d   = metrics.get("_diag", {})

    # 얼굴 너비/높이 (초록)
    _dot(img, d.get("l_cheek5"),  _C["face"])
    _dot(img, d.get("r_cheek5"),  _C["face"])
    _dot(img, d.get("brow_avg"),  _C["face"])
    _dot(img, d.get("chin_pt"),   _C["face"])
    _line(img, d.get("l_cheek5"), d.get("r_cheek5"), _C["face"])
    _line(img, d.get("brow_avg"), d.get("chin_pt"),  _C["face"])
    _label(img, d.get("l_cheek5"), "face_w(contour5)", _C["face"])
    _label(img, d.get("brow_avg"), "face_h", _C["face"])

    # 눈썹 → 삼등분 경계 (청록)
    _dot(img, d.get("l_brow"),  _C["brow"])
    _dot(img, d.get("r_brow"),  _C["brow"])
    _line(img, d.get("l_brow"), d.get("r_brow"), _C["brow"])
    _label(img, d.get("l_brow"), "brow_upper", _C["brow"])

    # 눈 (노랑) — 안쪽 눈구석이 미간 측정에 쓰임
    _dot(img, d.get("l_eye_inner"), _C["eye"])
    _dot(img, d.get("r_eye_inner"), _C["eye"])
    _dot(img, d.get("l_eye_outer"), _C["eye"])
    _dot(img, d.get("r_eye_outer"), _C["eye"])
    _line(img, d.get("l_eye_inner"), d.get("r_eye_inner"), _C["eye"], 3)
    _line(img, d.get("l_eye_outer"), d.get("l_eye_inner"), _C["eye"])
    _line(img, d.get("r_eye_inner"), d.get("r_eye_outer"), _C["eye"])
    _label(img, d.get("l_eye_inner"), "미간(inner)", _C["eye"])
    _label(img, d.get("l_eye_outer"), "눈꼬리(outer)", _C["eye"])

    # 코 (보라)
    _dot(img, d.get("nose_l"),   _C["nose"])
    _dot(img, d.get("nose_r"),   _C["nose"])
    _dot(img, d.get("nose_tip"), _C["nose"])
    _line(img, d.get("nose_l"), d.get("nose_r"), _C["nose"])
    _label(img, d.get("nose_l"), "콧볼(nose_left)", _C["nose"])

    # 입 너비 (파랑)
    _dot(img, d.get("mouth_l"), _C["mouth"])
    _dot(img, d.get("mouth_r"), _C["mouth"])
    _line(img, d.get("mouth_l"), d.get("mouth_r"), _C["mouth"])
    _label(img, d.get("mouth_l"), "입너비", _C["mouth"])

    # 입술 두께 (진파랑)
    _dot(img, d.get("upper_lip_top"), _C["lip"])
    _dot(img, d.get("upper_lip_bot"), _C["lip"])
    _dot(img, d.get("lower_lip_top"), _C["lip"])
    _dot(img, d.get("lower_lip_bot"), _C["lip"])
    _line(img, d.get("upper_lip_top"), d.get("upper_lip_bot"), _C["lip"])
    _line(img, d.get("lower_lip_top"), d.get("lower_lip_bot"), _C["lip"])
    _label(img, d.get("upper_lip_top"), "윗입술", _C["lip"])
    _label(img, d.get("lower_lip_top"), "아랫입술", _C["lip"])

    # 턱 (주황)
    _dot(img, d.get("jaw_l"), _C["jaw"])
    _dot(img, d.get("jaw_r"), _C["jaw"])
    _line(img, d.get("jaw_l"), d.get("chin_pt"), _C["jaw"])
    _line(img, d.get("jaw_r"), d.get("chin_pt"), _C["jaw"])
    _label(img, d.get("jaw_l"), "턱각도L(contour9)", _C["jaw"])

    # 범례
    legend = [
        ("초록: 얼굴 너비/높이",     _C["face"]),
        ("청록: 눈썹(삼등분 경계)",  _C["brow"]),
        ("노랑: 눈 미간·너비",       _C["eye"]),
        ("보라: 코",                 _C["nose"]),
        ("파랑: 입 너비",            _C["mouth"]),
        ("진파랑: 입술 두께",        _C["lip"]),
        ("주황: 턱 각도",            _C["jaw"]),
    ]
    for i, (text, col) in enumerate(legend):
        y = 50 + i * 20
        cv2.rectangle(img, (8, y - 12), (20, y), col, -1)
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 2)
        cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0),       1)

    cv2.putText(img, "Face++ 83p — 측정 점 진단", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
    return img


def make_comparison_image(mp_img: np.ndarray, fp_img: np.ndarray) -> np.ndarray:
    """두 이미지를 같은 높이로 맞춰 나란히 붙임."""
    h1, w1 = mp_img.shape[:2]
    h2, w2 = fp_img.shape[:2]
    target_h = max(h1, h2)

    def pad_height(img, target):
        dh = target - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    mp_img = pad_height(mp_img, target_h)
    fp_img = pad_height(fp_img, target_h)

    divider = np.full((target_h, 6, 3), 200, dtype=np.uint8)
    return np.hstack([mp_img, divider, fp_img])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 비교표 출력
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "face_wh_ratio":    "얼굴 가로/세로 비율",
    "forehead_ratio":   "이마 비율 [추정치]",
    "midface_ratio":    "중안부 비율",
    "lower_ratio":      "하안부 비율",
    "eye_gap_ratio":    "눈간격/눈너비 (오등분)",
    "nose_eye_ratio":   "코너비/눈간격",
    "mouth_nose_ratio": "입너비/코너비",
    "lip_ratio":        "윗입술/아랫입술 두께",
    "jaw_angle_deg":    "턱 각도 (도°)",
    "symmetry":         "좌우 대칭도",
}


def print_metrics_table(stem: str, mp_m: dict, fp_m: dict):
    print(f"\n{'='*72}")
    print(f"  {stem}")
    print(f"{'='*72}")
    header = f"{'지표':<28}  {'MediaPipe':>12}  {'Face++':>12}  {'차이(%)':>9}"
    print(header)
    print("-" * 72)
    for key, label in METRIC_LABELS.items():
        mv = mp_m.get(key)
        fv = fp_m.get(key)
        ms = f"{mv:.4f}" if mv is not None else "    N/A"
        fs = f"{fv:.4f}" if fv is not None else "    N/A"
        if mv is not None and fv is not None and fv != 0:
            diff = abs(mv - fv) / abs(fv) * 100
            ds = f"{diff:>8.1f}%"
        else:
            ds = "     N/A"
        print(f"  {label:<26}  {ms:>12}  {fs:>12}  {ds}")
    print("=" * 72)


def build_csv_rows(stem: str, mp_m: dict, fp_m: dict) -> list[list]:
    rows = []
    for key, label in METRIC_LABELS.items():
        mv = mp_m.get(key)
        fv = fp_m.get(key)
        if mv is not None and fv is not None and fv != 0:
            diff = f"{abs(mv - fv) / abs(fv) * 100:.1f}"
        else:
            diff = "N/A"
        rows.append([
            stem,
            label,
            f"{mv:.4f}" if mv is not None else "N/A",
            f"{fv:.4f}" if fv is not None else "N/A",
            diff,
        ])
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 메인
# ═══════════════════════════════════════════════════════════════════════════════

def collect_images(target: str) -> list[Path]:
    p = Path(target)
    if p.is_file():
        return [p]
    if p.is_dir():
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            imgs.extend(p.glob(ext))
        return sorted(imgs)
    print(f"[ERROR] 경로를 찾을 수 없습니다: {target}")
    sys.exit(1)


def process_image(image_path: Path, csv_writer) -> bool:
    stem = image_path.stem
    print(f"\n[처리] {image_path.name}")

    # ── MediaPipe ──
    mp_data, mp_err = extract_mediapipe(image_path)
    if mp_data is None:
        print(f"  [SKIP] {mp_err}")
        csv_writer.writerow([stem, "SKIP", mp_err, "", ""])
        return False

    # ── Face++ ──
    fp_data, fp_err = extract_facepp(image_path)
    if fp_data is None:
        print(f"  [SKIP] {fp_err}")
        csv_writer.writerow([stem, "SKIP", "", fp_err, ""])
        return False

    # ── 지표 계산 ──
    mp_metrics = metrics_from_mediapipe(mp_data)
    fp_metrics = metrics_from_facepp(fp_data)

    # ── 콘솔 출력 ──
    print_metrics_table(stem, mp_metrics, fp_metrics)

    # ── CSV 행 추가 ──
    for row in build_csv_rows(stem, mp_metrics, fp_metrics):
        csv_writer.writerow(row)

    # ── 시각화 (일반 + 진단) ──
    mp_vis   = draw_mediapipe(image_path, mp_data)
    fp_vis   = draw_facepp(image_path, fp_data)
    cmp_vis  = make_comparison_image(mp_vis, fp_vis)
    mp_diag  = draw_diagnostic_mediapipe(image_path, mp_metrics)
    fp_diag  = draw_diagnostic_facepp(image_path, fp_data, fp_metrics)
    cmp_diag = make_comparison_image(mp_diag, fp_diag)

    mp_out      = OUTPUT_DIR / f"{stem}_mediapipe.png"
    fp_out      = OUTPUT_DIR / f"{stem}_facepp.png"
    cmp_out     = OUTPUT_DIR / f"{stem}_compare.png"
    mp_diag_out = OUTPUT_DIR / f"{stem}_diag_mediapipe.png"
    fp_diag_out = OUTPUT_DIR / f"{stem}_diag_facepp.png"
    cmp_diag_out= OUTPUT_DIR / f"{stem}_diag_compare.png"

    def _imwrite(path: Path, arr: np.ndarray):
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError(f"imencode 실패: {path}")
        with open(str(path), "wb") as f:
            f.write(buf.tobytes())

    for path, arr in [
        (mp_out, mp_vis), (fp_out, fp_vis), (cmp_out, cmp_vis),
        (mp_diag_out, mp_diag), (fp_diag_out, fp_diag), (cmp_diag_out, cmp_diag),
    ]:
        _imwrite(path, arr)

    print(f"  저장: {mp_out.name}, {fp_out.name}, {cmp_out.name}")
    print(f"  진단: {mp_diag_out.name}, {fp_diag_out.name}, {cmp_diag_out.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="MediaPipe vs Face++ 랜드마크 비교")
    parser.add_argument("target", help="이미지 파일 또는 폴더 경로")
    args = parser.parse_args()

    images = collect_images(args.target)
    if not images:
        print("[ERROR] 이미지 파일이 없습니다.")
        sys.exit(1)

    print(f"총 {len(images)}장 처리 예정")

    csv_path = OUTPUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["사진", "지표", "MediaPipe", "Face++", "차이(%)"])

        ok = 0
        for img in images:
            if process_image(img, writer):
                ok += 1

    print(f"\n완료: {ok}/{len(images)}장 처리됨")
    print(f"CSV: {csv_path}")
    print(f"이미지: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
