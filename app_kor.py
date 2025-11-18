import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk
from scipy.signal import savgol_filter


# -------------------------
# 유틸 함수들
# -------------------------

def to_gray_np(uploaded_file):
    img = Image.open(uploaded_file).convert("L")   # grayscale
    return np.array(img)


def nlm_denoise(img, h_factor=1.15, patch_size=7, patch_distance=11):
    img_f = img.astype(np.float32) / 255.0
    sigma_est = np.mean(estimate_sigma(img_f, channel_axis=None))
    den = denoise_nl_means(
        img_f,
        h=h_factor * sigma_est,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
    )
    return (den * 255).astype(np.uint8)


def remove_central_beam(img, band_half=3):
    """
    img: 2D gray (numpy array)
    band_half: 중앙 빔 주변 몇 줄까지 같이 지울지
    """
    img2 = img.copy()
    h, w = img2.shape

    # 각 row의 평균 밝기 프로필
    profile = img2.mean(axis=1)
    r0 = int(np.argmax(profile))   # 가장 밝은 row = 빔 위치

    y0 = max(0, r0 - band_half)
    y1 = min(h, r0 + band_half)

    # 그 부분을 주변 median 값으로 채우기
    fill_val = np.median(img2)
    img2[y0:y1, :] = fill_val

    return img2, (y0, y1)

def get_anterior_lens_axis(lens_mask):
    """
    lens_mask : bool HxW (렌즈/홍채 blob)
    return:
        xs_lens : 길이 H, 각 y에서 anterior lens/iris의 x 좌표 (없으면 -1)
    """
    h, w = lens_mask.shape
    xs_lens = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        xs = np.where(lens_mask[y])[0]
        if xs.size == 0:
            continue
        # 가장 앞쪽(왼쪽) 픽셀 = anterior surface
        xs_lens[y] = xs.min()

    return xs_lens
def refine_axis_skip_center(xs_raw,
                            beam_band=None,
                            center_pad=5,
                            max_step=4,
                            min_len=40):
    """
    xs_raw    : cornea/lens axis (len=H), invalid <=0
    beam_band : (y0, y1)  # remove_central_beam에서 받은 빔 행 범위
    center_pad: 빔 주변으로 추가로 더 버릴 여유(px)
    max_step  : 연속 y에서 허용할 최대 x 변화
    min_len   : 최소 segment 길이
    """
    xs = xs_raw.astype(float).copy()
    xs[xs <= 0] = np.nan

    h = len(xs)

    # 1) 중앙 빔 주변 y구간은 아예 제외
    if beam_band is not None:
        y0, y1 = beam_band
        y0 = max(0, y0 - center_pad)
        y1 = min(h, y1 + center_pad)
        xs[y0:y1] = np.nan   # 가운데는 곡면 추정에서 완전히 제거

    # 2) 남은 부분들 중에서 Δx 작은 연속 segment만 찾기
    ys = np.where(~np.isnan(xs))[0]
    if ys.size < 3:
        return xs  # fallback

    xs_valid = xs[ys]
    dx = np.abs(np.diff(xs_valid))
    good = dx <= max_step

    segs = []
    start = 0
    for i, g in enumerate(good):
        if not g:
            segs.append((start, i))
            start = i + 1
    segs.append((start, len(xs_valid) - 1))

    # 최소 길이 조건
    segs = [s for s in segs if (s[1] - s[0] + 1) >= min_len]
    if not segs:
        return xs

    # 가장 긴 segment 선택
    s, e = max(segs, key=lambda t: t[1] - t[0])

    ys_main = ys[s:e+1]
    xs_main = xs_valid[s:e+1]

    xs_new = np.full_like(xs, np.nan, dtype=float)
    xs_new[ys_main] = xs_main
    return xs_new

# -------------------------
# cornea / lens mask 추출
# -------------------------

def get_cornea_lens_masks(img, k=2.0, min_area_ratio=0.001):
    """
    img : NLM 후 그레이 이미지 (beam 제거 전)
    k   : thr = mean + k * std
    min_area_ratio : 너무 작은 blob 제거용 (이미지 전체의 비율)

    return:
        cornea_mask, lens_mask (bool HxW)
    """
    h, w = img.shape
    mean = img.mean()
    std  = img.std()
    thr  = mean + k * std

    binary = img > thr

    lbl = label(binary)
    regs = regionprops(lbl)

    if len(regs) < 2:
        raise RuntimeError("밝은 구조 blob을 2개 이상 찾지 못했습니다. (k 값을 줄여보세요)")

    min_area = h * w * min_area_ratio

    left_candidates = []
    right_candidates = []

    for r in regs:
        if r.area < min_area:
            continue
        y0, x0, y1, x1 = r.bbox
        cx = (x0 + x1) / 2.0
        if cx < w / 2:
            left_candidates.append((r.area, r))
        else:
            right_candidates.append((r.area, r))

    if not left_candidates or not right_candidates:
        raise RuntimeError("왼쪽(각막) 또는 오른쪽(렌즈) blob을 찾지 못했습니다.")

    cornea_r = max(left_candidates, key=lambda x: x[0])[1]
    lens_r   = max(right_candidates, key=lambda x: x[0])[1]

    cornea_mask = (lbl == cornea_r.label)
    lens_mask   = (lbl == lens_r.label)

    return cornea_mask, lens_mask


# -------------------------
# 1D curve smoothing helper
# -------------------------

def smooth_1d(x, window=15):
    """
    간단 moving-average smoothing.
    window: 홀수가 자연스럽고, 클수록 더 매끈해짐.
    """
    if window < 3:
        return x
    if window % 2 == 0:
        window += 1
    k = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, k, mode="same")


# -------------------------
# 각막 / 렌즈 호 보간 + 스무딩
# -------------------------

def fit_cornea_arc_from_mask(mask_cornea, smooth_win=15):
    """
    mask_cornea: bool HxW
    return: xs_cornea[y] = posterior cornea x (float, 보간 + 스무딩)
    """
    h, w = mask_cornea.shape
    xs_arc = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask_cornea[y])[0]
        if len(xs) > 0:
            xs_arc[y] = xs.max()  # posterior 쪽

    valid = ~np.isnan(xs_arc)
    if valid.sum() < 2:
        raise RuntimeError("cornea arc를 추정할 수 없습니다.")

    ys = np.arange(h)
    xs_interp = np.interp(ys, ys[valid], xs_arc[valid])

    if smooth_win and smooth_win > 1:
        xs_interp = smooth_1d(xs_interp, window=smooth_win)

    return xs_interp


def fit_lens_arc_from_mask(mask_lens, smooth_win=15):
    """
    mask_lens: bool HxW
    return: xs_lens[y] = anterior lens/iris x (float, 보간 + 스무딩)
    """
    h, w = mask_lens.shape
    xs_arc = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask_lens[y])[0]
        if len(xs) > 0:
            xs_arc[y] = xs.min()  # anterior 쪽

    valid = ~np.isnan(xs_arc)
    if valid.sum() < 2:
        raise RuntimeError("lens arc를 추정할 수 없습니다.")

    ys = np.arange(h)
    xs_interp = np.interp(ys, ys[valid], xs_arc[valid])

    if smooth_win and smooth_win > 1:
        xs_interp = smooth_1d(xs_interp, window=smooth_win)

    return xs_interp

def get_posterior_cornea_mask(cornea_mask, dilate_r=1):
    """
    cornea_mask : bool HxW (앞/뒷면이 다 들어있을 수 있음)
    return:
        post_mask : bool HxW (각막 '뒷면'만을 따라가는 얇은 mask)
        xs_post   : 길이 H, 각 y에서 posterior x (없으면 -1)
    """
    h, w = cornea_mask.shape
    post = np.zeros_like(cornea_mask, dtype=bool)
    xs_post = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        xs = np.where(cornea_mask[y])[0]
        if xs.size == 0:
            continue
        # 가장 안쪽(뒤쪽, 오른쪽) 픽셀 = posterior surface
        x_p = xs.max()
        post[y, x_p] = True
        xs_post[y] = x_p

    if dilate_r > 0:
        post = dilation(post, disk(dilate_r))

    return post, xs_post
# -------------------------
# 호-호 + 두 변 = annulus 조각 mask
# -------------------------
def build_ac_mask_with_chords(img_shape,
                              xs_cornea_raw,
                              xs_lens_raw,
                              min_width=5):
    """
    img_shape      : (H, W)
    xs_cornea_raw  : 길이 H, posterior cornea x (없으면 -1/0/NaN)
    xs_lens_raw    : 길이 H, anterior lens/iris x (없으면 -1/0/NaN)

    반환:
        ac_mask : bool HxW

    폴리곤 구조:
      - cornea 호:   위 → 아래 (각막 posterior 전체)
      - 아래 chord: cornea_bottom → lens_bottom (사선)
      - lens 호:     아래 → 위 (렌즈/홍채 전체)
      - 위 chord:   lens_top → cornea_top (사선)
    """
    h, w = img_shape

    xs_c = xs_cornea_raw.astype(float).copy()
    xs_l = xs_lens_raw.astype(float).copy()

    # 유효하지 않은 값들은 NaN 처리
    xs_c[xs_c <= 0] = np.nan
    xs_l[xs_l <= 0] = np.nan

    ys_all = np.arange(h)

    # 각 곡면이 실제로 존재하는 y 인덱스
    ys_c = np.where(~np.isnan(xs_c))[0]
    ys_l = np.where(~np.isnan(xs_l))[0]

    if ys_c.size < 2:
        raise RuntimeError("각막 posterior 곡면이 너무 짧습니다.")
    if ys_l.size < 2:
        raise RuntimeError("렌즈/홍채 곡면이 너무 짧습니다.")

    # 폭이 너무 좁지 않은지 안전 체크 (겹치는 y에서만)
    ys_overlap = np.intersect1d(ys_c, ys_l)
    if ys_overlap.size > 0:
        width = xs_l[ys_overlap] - xs_c[ys_overlap]
        if np.all(width <= min_width):
            raise RuntimeError("AC 폭이 너무 좁습니다. min_width나 threshold를 조정해보세요.")

    # --- cornea 곡면: 위→아래 ---
    ys_c_sorted = np.sort(ys_c)
    xs_c_sorted = xs_c[ys_c_sorted]
    pts_cornea = np.stack([xs_c_sorted, ys_c_sorted], axis=1).astype(np.int32)

    # --- lens 곡면: 위→아래 (나중에 아래→위로 쓸 거라 reverse) ---
    ys_l_sorted = np.sort(ys_l)
    xs_l_sorted = xs_l[ys_l_sorted]
    pts_lens = np.stack([xs_l_sorted, ys_l_sorted], axis=1).astype(np.int32)

    # 각막 위/아래 끝점, 렌즈 위/아래 끝점
    cornea_top    = pts_cornea[0]         # (x_c_top, y_c_top)
    cornea_bottom = pts_cornea[-1]        # (x_c_bot, y_c_bot)
    lens_top      = pts_lens[0]           # (x_l_top, y_l_top)
    lens_bottom   = pts_lens[-1]          # (x_l_bot, y_l_bot)

    # 아래쪽에서 폭이 너무 작으면 옵션으로 막을 수도 있음
    if (lens_bottom[0] - cornea_bottom[0]) <= min_width:
        # 필요하면 여기서만 살짝 min_width 보정해도 됨
        pass

    # 폴리곤 만들기:
    #   1) cornea 호 (위→아래)
    #   2) 아래 chord: cornea_bottom → lens_bottom
    #   3) lens 호 (아래→위)
    #   4) 위 chord: lens_top → cornea_top
    poly_points = []

    # 1) cornea 호
    poly_points.extend(pts_cornea.tolist())

    # 2) 아래 chord (각막 아래끝 → 렌즈 아래끝)
    poly_points.append(lens_bottom.tolist())

    # 3) lens 호: 아래→위
    poly_points.extend(pts_lens[::-1].tolist())

    # 4) 위 chord (렌즈 위끝 → 각막 위끝)
    poly_points.append(cornea_top.tolist())

    polygon = np.array(poly_points, dtype=np.int32)

    mask = np.zeros(img_shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)

    return mask.astype(bool)

# -------------------------
# cell detection & overlay
# -------------------------

def detect_cells(binary, area_min=2, area_max=30, circ_min=0.4):
    lbl = label(binary)
    regs = regionprops(lbl)
    cells = []
    for r in regs:
        area = r.area
        if area < area_min or area > area_max:
            continue
        perim = r.perimeter
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < circ_min:
            continue
        cells.append(r)
    return cells, lbl


def overlay_cells_on_roi(img_roi, cells, mask_roi=None):
    base = img_roi.copy()
    if mask_roi is not None:
        base = base.copy()
        base[~mask_roi] = 0

    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for r in cells:
        cy, cx = r.centroid
        cx, cy = int(cx), int(cy)
        cv2.circle(bgr, (cx, cy), 3, (0, 0, 255), -1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def overlay_cells_on_full(img_gray, cells, x0, y0, roi_shape):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for r in cells:
        cy, cx = r.centroid
        cx_f = int(cx) + x0
        cy_f = int(cy) + y0
        cv2.circle(bgr, (cx_f, cy_f), 3, (0, 0, 255), -1)

    h_roi, w_roi = roi_shape
    cv2.rectangle(bgr, (x0, y0), (x0 + w_roi, y0 + h_roi), (0, 255, 0), 1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def overlay_ac_mask_on_full(img_gray, ac_mask):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ac_uint = (ac_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(ac_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="IOLMaster AC Cell Quantification (Annulus ROI)", layout="wide")
st.title("IOLMaster Anterior Chamber Cell Quantification (Annulus-like ROI)")

uploaded = st.file_uploader(
    "IOLMaster B-scan 이미지 업로드 (PNG/JPG/TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is None:
    st.info("이미지를 업로드하면 분석이 시작됩니다.")
    st.stop()

img_orig = to_gray_np(uploaded)
h_img, w_img = img_orig.shape

st.sidebar.header("1. NL-means Denoising")
h_factor = st.sidebar.slider("h factor (noise level)", 0.5, 3.0, 1.15, 0.05)
patch_size = st.sidebar.slider("patch size", 3, 11, 7, 2)
patch_distance = st.sidebar.slider("patch distance", 5, 21, 11, 2)

st.sidebar.header("2. Beam removal")
beam_half = st.sidebar.slider("Beam half thickness (rows)", 1, 30, 7)

st.sidebar.header("3. AC ROI (annulus)")
k_val = st.sidebar.slider("Brightness threshold k (cornea/lens)", 0.5, 4.0, 0.8, 0.1)
margin = st.sidebar.slider("Margin from arcs (px)", 0, 10, 1)
min_width = st.sidebar.slider("Min AC width per row (px)", 1, 40, 5)
arc_smooth = st.sidebar.slider("Arc smoothing window (rows)", 1, 51, 15, 2)

st.sidebar.header("4. Threshold & Cell 조건")
manual_T = st.sidebar.checkbox("Manual threshold 사용 (AC 내부)", value=False)
T_manual = st.sidebar.slider("Threshold (0-255)", 0, 255, 30) if manual_T else None
area_min = st.sidebar.slider("Min area (px²)", 1, 100, 2)
area_max = st.sidebar.slider("Max area (px²)", 5, 300, 30)
circ_min = st.sidebar.slider("Min circularity", 0.0, 1.0, 0.4, 0.05)

step = st.radio(
    "어느 단계를 볼까요?",
    ["Original", "NLM", "Beam removed", "AC ROI", "Thresholded", "AC ROI + Cells", "Full B-scan + Cells"],
    horizontal=True,
)

# -------------------------
# 파이프라인 실행
# -------------------------

# 1) NLM
img_nlm = nlm_denoise(
    img_orig,
    h_factor=h_factor,
    patch_size=patch_size,
    patch_distance=patch_distance,
)

# 2) 중앙 빔 제거 (먼저 해서 beam_band 확보)
img_beam_removed, beam_band = remove_central_beam(
    img_nlm,
    band_half=beam_half,
)
beam_y0, beam_y1 = beam_band

# 3) AC ROI용 cornea / lens mask (beam 제거 전 이미지에서)
error_msg = None
ac_mask = None
img_roi = None
yy0 = yy1 = x0 = x1 = None
cells = []
binary = None
overlay_roi = None
overlay_full = None

try:
    cornea_mask_raw, lens_mask = get_cornea_lens_masks(img_nlm, k=k_val)

    cornea_post_mask, xs_cornea = get_posterior_cornea_mask(cornea_mask_raw, dilate_r=1)
    xs_lens = get_anterior_lens_axis(lens_mask)

    xs_cornea = refine_axis_skip_center(
    xs_cornea,
    beam_band=beam_band,   # ⬅⬅ 여기서 방금 만든 beam_band 사용
    center_pad=5,
    max_step=4,
    min_len=40,
)

    ac_mask = build_ac_mask_with_chords(
    img_shape=img_nlm.shape,
    xs_cornea_raw=xs_cornea,
    xs_lens_raw=xs_lens,
    min_width=min_width,
)
    ys, xs = np.where(ac_mask)
    yy0 = ys.min()
    yy1 = ys.max() + 1
    x0  = xs.min()
    x1  = xs.max() + 1

except Exception as e:
    error_msg = str(e)

# 4) Threshold & cell detection
if ac_mask is not None:
    img_roi = img_beam_removed[yy0:yy1, x0:x1]
    mask_roi = ac_mask[yy0:yy1, x0:x1]

    if manual_T:
        T = T_manual
    else:
        vals_in_ac = img_roi[mask_roi]
        if len(vals_in_ac) == 0:
            T = threshold_otsu(img_roi)
        else:
            T = threshold_otsu(vals_in_ac)

    full_binary = (img_roi > T).astype(np.uint8) * 255
    binary = np.zeros_like(full_binary)
    binary[mask_roi] = full_binary[mask_roi]

    cells, _ = detect_cells(
        binary,
        area_min=area_min,
        area_max=area_max,
        circ_min=circ_min,
    )

    overlay_roi = overlay_cells_on_roi(img_roi, cells, mask_roi=mask_roi)
    overlay_full = overlay_cells_on_full(
        img_beam_removed,
        cells,
        x0,
        yy0,
        roi_shape=img_roi.shape,
    )

# -------------------------
# 화면 출력
# -------------------------

st.write(f"이미지 크기: **{w_img} x {h_img}** (W x H)")

if error_msg:
    st.warning(f"AC ROI 자동 탐지 경고: {error_msg}")

if img_roi is not None:
    st.write(f"AC ROI 좌표 (x0,x1,y0,y1): ({x0}, {x1}, {yy0}, {yy1})")
    st.write(f"검출된 cell 수: **{len(cells)}**")
else:
    st.write("AC ROI를 찾지 못했습니다. k, margin, min_width 등을 조정해보세요.")

if step == "Original":
    st.image(img_orig, caption="Original B-scan", clamp=True)

elif step == "NLM":
    st.image(img_nlm, caption="NLM denoised", clamp=True)

elif step == "Beam removed":
    st.image(
        img_beam_removed,
        caption="NLM denoised + central beam removed",
        clamp=True,
    )

elif step == "AC ROI":
    if ac_mask is not None:
        ac_overlay = overlay_ac_mask_on_full(img_orig, ac_mask)
        st.image(ac_overlay, caption="AC annulus-like ROI (green contour)", clamp=True)
        if img_roi is not None:
            st.image(img_roi, caption="AC ROI crop (beam removed)", clamp=True)
    else:
        st.error("AC ROI가 정의되지 않았습니다.")

elif step == "Thresholded":
    if binary is not None:
        st.image(binary, caption="Thresholded AC ROI (AC band only)", clamp=True)
    else:
        st.error("Threshold 이미지를 생성할 수 없습니다.")

elif step == "AC ROI + Cells":
    if overlay_roi is not None:
        st.image(
            overlay_roi,
            caption=f"AC ROI + red cells (count={len(cells)})",
            clamp=True,
        )
    else:
        st.error("AC ROI overlay를 생성할 수 없습니다.")

elif step == "Full B-scan + Cells":
    if overlay_full is not None and ac_mask is not None:
        ac_overlay = overlay_ac_mask_on_full(img_beam_removed, ac_mask)
        st.image(
            ac_overlay,
            caption="Full B-scan with AC annulus ROI (green)",
            clamp=True,
        )
        st.image(
            overlay_full,
            caption="Full B-scan with AC cells (red dots)",
            clamp=True,
        )
    else:
        st.error("전체 이미지 overlay를 생성할 수 없습니다.")
