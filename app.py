import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

# -------------------------
# Utility functions
# -------------------------
def to_gray_np(uploaded_file):
    img = Image.open(uploaded_file).convert("L")   # grayscale
    return np.array(img)

def autocrop_vertical_white(img, white_thr=250):
    """
    img       : 2D gray (H x W)
    white_thr : row_mean >= white_thr 는 white background 로 간주
    return    : (cropped_img, (y0, y1))
    """
    h, w = img.shape
    row_mean = img.mean(axis=1)

    # '유효한 row' = B-scan 포함된 줄
    valid = np.where(row_mean < white_thr)[0]
    if valid.size == 0:
        # 전체가 흰 배경이면 그대로 반환
        return img, (0, h)

    # 연속 구간으로 쪼개기
    segments = []
    start = prev = valid[0]
    for y in valid[1:]:
        if y == prev + 1:
            prev = y
        else:
            segments.append((start, prev))
            start = prev = y
    segments.append((start, prev))

    # 가장 긴 연속 구간만 선택
    y0, y1 = max(segments, key=lambda s: s[1] - s[0])

    # crop — y1 inclusive → +1
    return img[y0:y1+1, :], (y0, y1+1)

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
    band_half: remove beam ± several rows
    """
    img2 = img.copy()
    h, w = img2.shape

    # bright beam row detection
    profile = img2.mean(axis=1)
    r0 = int(np.argmax(profile))

    y0 = max(0, r0 - band_half)
    y1 = min(h, r0 + band_half)

    fill_val = np.median(img2)
    img2[y0:y1, :] = fill_val

    return img2, (y0, y1)

def get_anterior_lens_axis(lens_mask):
    """
    lens_mask: bool HxW
    Returns anterior lens/iris x for each row (min x)
    """
    h, w = lens_mask.shape
    xs_lens = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        xs = np.where(lens_mask[y])[0]
        if xs.size == 0:
            continue
        xs_lens[y] = xs.min()

    return xs_lens

def refine_axis_skip_center(xs_raw,
                            beam_band=None,
                            center_pad=5,
                            max_step=4,
                            min_len=40,
                            min_center_width=24):
    """
    Remove axis points near beam + choose longest stable arc.

    - beam_band 주위는 center_pad 만큼 확장해서 버리고
    - 그 폭이 너무 좁으면 min_center_width 만큼은 무조건 버리도록 보정
    """
    xs = xs_raw.astype(float).copy()
    xs[xs <= 0] = np.nan

    h = len(xs)

    # 1) 중앙 빔 주변 y구간 넉넉하게 제외
    if beam_band is not None:
        y0, y1 = beam_band          # 원래 빔 범위 (band 좌표계)
        mid = (y0 + y1) // 2        # 중앙
        half = (y1 - y0) // 2 + center_pad

        # 최소 폭 보장 (예: 최소 24px 정도는 항상 버리도록)
        half = max(half, min_center_width // 2)

        y0 = max(0, mid - half)
        y1 = min(h, mid + half)

        xs[y0:y1] = np.nan   # 이 중앙 구간은 곡선 추정에서 완전히 제거

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

    segs = [s for s in segs if (s[1] - s[0] + 1) >= min_len]
    if not segs:
        return xs

    s, e = max(segs, key=lambda t: t[1] - t[0])
    ys_main = ys[s:e+1]
    xs_main = xs_valid[s:e+1]

    xs_new = np.full_like(xs, np.nan, dtype=float)
    xs_new[ys_main] = xs_main
    return xs_new

# -------------------------
# Cornea/Lens mask extraction
# -------------------------
def get_cornea_lens_masks(img, k=2.0, min_area_ratio=0.001, dx_min_ratio=0.02):
    """
    Detect 2 bright blobs (cornea & lens) by sorting blobs by x-position
    + area heuristic.

    - cornea: 가장 왼쪽 큰 blob
    - lens: cornea 오른쪽에 있는 blob 중 area가 가장 큰 것
    """
    h, w = img.shape
    mean = img.mean()
    std  = img.std()
    thr  = mean + k * std

    binary = img > thr
    lbl = label(binary)
    regs = regionprops(lbl)

    if len(regs) < 2:
        raise RuntimeError("Could not detect ≥2 bright blobs. Try lowering k.")

    min_area = h * w * min_area_ratio

    # keep only large blobs
    big = [r for r in regs if r.area >= min_area]
    if len(big) < 2:
        raise RuntimeError("Not enough large bright blobs (need ≥2).")

    # x0 기준 정렬
    big_sorted = sorted(big, key=lambda r: r.bbox[1])

    # 1) cornea = 가장 왼쪽 큰 blob
    cornea_r = big_sorted[0]

    # 2) lens 후보 = cornea 보다 오른쪽에 있는 blob들 중 area가 큰 것
    dx_min = int(w * dx_min_ratio)  # cornea에서 최소 이 정도는 떨어지도록
    cornea_x0 = cornea_r.bbox[1]

    lens_candidates = [
        r for r in big
        if (r.bbox[1] - cornea_x0) >= dx_min
    ]

    if not lens_candidates:
        # fallback: 예전처럼 두 번째 blob 사용
        if len(big_sorted) < 2:
            raise RuntimeError("No lens candidate blob found.")
        lens_r = big_sorted[1]
    else:
        # area 가장 큰 blob을 lens로
        lens_r = max(lens_candidates, key=lambda r: r.area)

    cornea_mask = (lbl == cornea_r.label)
    lens_mask   = (lbl == lens_r.label)

    return cornea_mask, lens_mask

# -------------------------
# 1D smoothing
# -------------------------
def smooth_1d(x, window=15):
    if window < 3:
        return x
    if window % 2 == 0:
        window += 1
    k = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, k, mode="same")

# -------------------------
# Arc interpolation
# -------------------------
def fit_cornea_arc_from_mask(mask_cornea, smooth_win=15):
    """
    Posterior cornea arc extraction
    """
    h, w = mask_cornea.shape
    xs_arc = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask_cornea[y])[0]
        if xs.size > 0:
            xs_arc[y] = xs.max()

    valid = ~np.isnan(xs_arc)
    if valid.sum() < 2:
        raise RuntimeError("Cannot estimate corneal arc.")

    ys = np.arange(h)
    xs_interp = np.interp(ys, ys[valid], xs_arc[valid])

    if smooth_win > 1:
        xs_interp = smooth_1d(xs_interp, window=smooth_win)

    return xs_interp

def fit_lens_arc_from_mask(mask_lens, smooth_win=15):
    """
    Anterior lens/iris arc extraction
    """
    h, w = mask_lens.shape
    xs_arc = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(mask_lens[y])[0]
        if xs.size > 0:
            xs_arc[y] = xs.min()

    valid = ~np.isnan(xs_arc)
    if valid.sum() < 2:
        raise RuntimeError("Cannot estimate lens arc.")

    ys = np.arange(h)
    xs_interp = np.interp(ys, ys[valid], xs_arc[valid])

    if smooth_win > 1:
        xs_interp = smooth_1d(xs_interp, window=smooth_win)

    return xs_interp

def get_posterior_cornea_mask(cornea_mask, dilate_r=1):
    """
    Extract posterior-most corneal boundary
    """
    h, w = cornea_mask.shape
    post = np.zeros_like(cornea_mask, dtype=bool)
    xs_post = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        xs = np.where(cornea_mask[y])[0]
        if xs.size == 0:
            continue
        x_p = xs.max()
        post[y, x_p] = True
        xs_post[y] = x_p

    if dilate_r > 0:
        post = dilation(post, disk(dilate_r))

    return post, xs_post

# -------------------------
# AC “annulus-like” ROI polygon
# -------------------------
def build_ac_mask_with_chords(img_shape,
                              xs_cornea_raw,
                              xs_lens_raw,
                              min_width=5):
    """
    Construct polygon: cornea arc + bottom chord + lens arc + top chord.
    """
    h, w = img_shape
    xs_c = xs_cornea_raw.astype(float).copy()
    xs_l = xs_lens_raw.astype(float).copy()

    xs_c[xs_c <= 0] = np.nan
    xs_l[xs_l <= 0] = np.nan

    ys_c = np.where(~np.isnan(xs_c))[0]
    ys_l = np.where(~np.isnan(xs_l))[0]

    if ys_c.size < 2:
        raise RuntimeError("Corneal arc too short.")
    if ys_l.size < 2:
        raise RuntimeError("Lens arc too short.")

    ys_overlap = np.intersect1d(ys_c, ys_l)
    if ys_overlap.size > 0:
        width = xs_l[ys_overlap] - xs_c[ys_overlap]
        if np.all(width <= min_width):
            raise RuntimeError("AC width too narrow. Increase min_width or adjust thresholds.")

    # cornea arc (top → bottom)
    ys_c_sorted = np.sort(ys_c)
    pts_cornea = np.stack([xs_c[ys_c_sorted], ys_c_sorted], axis=1).astype(np.int32)

    # lens arc (bottom → top)
    ys_l_sorted = np.sort(ys_l)
    pts_lens = np.stack([xs_l[ys_l_sorted], ys_l_sorted], axis=1).astype(np.int32)

    cornea_top = pts_cornea[0]
    cornea_bottom = pts_cornea[-1]
    lens_top = pts_lens[0]
    lens_bottom = pts_lens[-1]

    poly_points = []
    poly_points.extend(pts_cornea.tolist())
    poly_points.append(lens_bottom.tolist())
    poly_points.extend(pts_lens[::-1].tolist())
    poly_points.append(cornea_top.tolist())

    polygon = np.array(poly_points, dtype=np.int32)

    mask = np.zeros(img_shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)

    return mask.astype(bool)

# -------------------------
# Cell detection
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
        cv2.circle(bgr, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_cells_on_full(img_gray, cells, x0, y0, roi_shape):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for r in cells:
        cy, cx = r.centroid
        cx_f = int(cx) + x0
        cy_f = int(cy) + y0
        cv2.circle(bgr, (cx_f, cy_f), 3, (0, 0, 255), -1)

    h_roi, w_roi = roi_shape
    cv2.rectangle(bgr, (x0, y0), (x0 + w_roi, y0 + h_roi), (0, 255, 0), 1)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_ac_mask_on_full(img_gray, ac_mask_full):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ac_uint = (ac_mask_full.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(ac_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="IOLMaster AC Cell Quantification (Annulus ROI)", layout="wide")
st.title("IOLMaster Anterior Chamber Cell Quantification (Annulus-like ROI)")

uploaded = st.file_uploader(
    "Upload IOLMaster B-scan image (PNG/JPG/TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is None:
    st.info("Upload an image to begin analysis.")
    st.stop()

img_orig = to_gray_np(uploaded)
# 작업용 이미지 (필요시 위/아래 white 영역 컷)
img_orig, (crop_y0, crop_y1) = autocrop_vertical_white(img_orig)
img_work = img_orig.copy()
h_img, w_img = img_orig.shape

st.sidebar.header("1. NL-means Denoising")
h_factor = st.sidebar.slider("h factor (noise level)", 0.5, 3.0, 1.15, 0.05)
patch_size = st.sidebar.slider("patch size", 3, 11, 7, 2)
patch_distance = st.sidebar.slider("patch distance", 5, 21, 11, 2)

st.sidebar.header("2. Beam Removal")
beam_half = st.sidebar.slider("Beam half thickness (rows)", 1, 30, 11)

st.sidebar.header("3. AC ROI (annulus)")
k_val = st.sidebar.slider("Brightness threshold k (cornea/lens)", 0.5, 4.0, 0.8, 0.1)
min_width = st.sidebar.slider("Min AC width per row (px)", 1, 40, 5)
arc_smooth = st.sidebar.slider("Arc smoothing window (rows)", 1, 51, 15, 2)

st.sidebar.header("4. Threshold & Cell Detection")
manual_T = st.sidebar.checkbox("Use manual threshold", value=False)
T_manual = st.sidebar.slider("Threshold (0–255)", 0, 255, 30) if manual_T else None
area_min = st.sidebar.slider("Min area (px²)", 1, 100, 2)
area_max = st.sidebar.slider("Max area (px²)", 5, 300, 30)
circ_min = st.sidebar.slider("Min circularity", 0.0, 1.0, 0.4, 0.05)

step = st.radio(
    "View stage:",
    ["Original", "NLM", "Beam removed", "AC ROI", "Thresholded", "AC ROI + Cells", "Full B-scan + Cells"],
    horizontal=True,
)

# -------------------------
# Pipeline execution
# -------------------------

# 1) NLM
img_nlm = nlm_denoise(
    img_work,
    h_factor=h_factor,
    patch_size=patch_size,
    patch_distance=patch_distance,
)

# 2) Beam removal (전체 높이 기준)
img_beam_removed, beam_band_global = remove_central_beam(
    img_nlm,
    band_half=beam_half,
)

beam_y0_global, beam_y1_global = beam_band_global

error_msg = None
ac_mask_full = None
ac_mask_band = None
img_roi = None
yy0_global = yy1_global = x0 = x1 = None
cells = []
binary = None
overlay_roi = None
overlay_full = None

# -------------------------
# ✅ FIXED X + ANATOMICAL AC DETECTION
# -------------------------

error_msg = None
cells = []
binary = None
overlay_roi = None
overlay_full = None
ac_mask_full = None

H, W = img_beam_removed.shape

# ✅ (1) X = 140~300 고정
x0 = 140
x1 = 300

# ✅ (2) Y = 전체
yy0_global = 0
yy1_global = H

# ✅ (3) X-고정 ROI 안에서 AC 검출용 이미지
img_xband_nlm = img_nlm[:, x0:x1]
img_xband_beam = img_beam_removed[:, x0:x1]

try:
    # ✅ cornea / lens detection (X-band 내부)
    cornea_mask_raw, lens_mask = get_cornea_lens_masks(img_xband_nlm, k=k_val)

    cornea_post_mask, xs_cornea = get_posterior_cornea_mask(cornea_mask_raw, dilate_r=1)
    xs_lens = get_anterior_lens_axis(lens_mask)

    # ✅ beam 주변 axis 제거
    band_beam_band = None
    if beam_y1_global > 0 and beam_y0_global < H:
        band_beam_band = (beam_y0_global, beam_y1_global)

    xs_cornea_refined = refine_axis_skip_center(
        xs_cornea,
        beam_band=band_beam_band,
        center_pad=5,
        max_step=4,
        min_len=40,
    )

    # ✅ (4) annulus polygon AC mask (X-band 좌표계)
    ac_mask_band = build_ac_mask_with_chords(
        img_shape=img_xband_nlm.shape,
        xs_cornea_raw=xs_cornea_refined,
        xs_lens_raw=xs_lens,
        min_width=min_width,
    )

    # ✅ (global AC mask 복원)
    ac_mask_full = np.zeros_like(img_beam_removed, dtype=bool)
    ac_mask_full[:, x0:x1] = ac_mask_band

    # ✅ (5) AC ROI crop
    ys, xs = np.where(ac_mask_full)
    yy0_global = ys.min()
    yy1_global = ys.max() + 1
    x0_final = xs.min()
    x1_final = xs.max() + 1

    img_roi = img_beam_removed[yy0_global:yy1_global, x0_final:x1_final]
    mask_roi = ac_mask_full[yy0_global:yy1_global, x0_final:x1_final]

    # ✅ Threshold (AC 내부 기준)
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

    # ✅ (5) Cell detection (AC 내부)
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
        x0_final,
        yy0_global,
        roi_shape=img_roi.shape,
    )

except Exception as e:
    error_msg = str(e)

# -------------------------
# Display
# -------------------------
st.write(f"Image size (after vertical crop): **{w_img} x {h_img}** (W x H)")

if error_msg:
    st.warning(f"AC ROI detection warning: {error_msg}")

if img_roi is not None:
    st.write(f"AC ROI coords (x0,x1,y0,y1): ({x0_final}, {x1_final}, {yy0_global}, {yy1_global})")
    st.write(f"Detected cell count: **{len(cells)}**")
else:
    st.write("AC ROI could not be detected. Adjust k or min_width.")

if step == "Original":
    st.image(img_orig, caption="Original (vertically autocropped)", clamp=True)

elif step == "NLM":
    st.image(img_nlm, caption="NLM denoised", clamp=True)

elif step == "Beam removed":
    st.image(img_beam_removed, caption="Beam removed (whole image)", clamp=True)

elif step == "AC ROI":
    if ac_mask_full is not None:
        ac_overlay = overlay_ac_mask_on_full(img_beam_removed, ac_mask_full)
        st.image(ac_overlay, caption="AC annulus-like ROI (green contour) on full B-scan", clamp=True)
        if img_roi is not None:
            st.image(img_roi, caption="AC ROI crop (beam removed)", clamp=True)
    else:
        st.error("AC ROI not defined.")

elif step == "Thresholded":
    if binary is not None:
        st.image(binary, caption="Thresholded AC ROI", clamp=True)
    else:
        st.error("No threshold image available.")

elif step == "AC ROI + Cells":
    if overlay_roi is not None:
        st.image(
            overlay_roi,
            caption=f"AC ROI + detected cells (red dots)  count={len(cells)}",
            clamp=True,
        )
    else:
        st.error("ROI overlay unavailable.")

elif step == "Full B-scan + Cells":
    if overlay_full is not None and ac_mask_full is not None:
        ac_overlay = overlay_ac_mask_on_full(img_beam_removed, ac_mask_full)
        st.image(ac_overlay, caption="Full scan with AC annulus ROI (green)", clamp=True)
        st.image(overlay_full, caption="Full scan with detected cells (red)", clamp=True)
    else:
        st.error("Full image overlay unavailable.")
