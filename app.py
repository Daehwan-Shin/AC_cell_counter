import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import white_tophat, disk
from skimage.feature import blob_log

# =============================================================
# 1. Resolution-normalized parameters
#    해상도 독립성을 위해 모든 픽셀 단위 파라미터를 이미지 크기로부터 파생.
# =============================================================
def get_norm_params(H, W):
    ref = min(H, W)
    return dict(
        beam_k=3.0,
        beam_half_max=max(4, int(H * 0.04)),
        x_band_fallback=(0.20, 0.60),
        x_band_margin=0.05,
        min_area_ratio=0.0008,
        dx_min_ratio=0.02,
        poly_degree=2,
        poly_iter=5,
        poly_resid_k=2.5,
        beam_pad=max(3, int(H * 0.012)),
        cornea_thr_k=1.5,
        cell_disk=max(2, int(ref * 0.006)),
    )

# =============================================================
# 2. I/O + preprocessing
# =============================================================
def to_gray_np(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    return np.array(img)

def autocrop_vertical_white(img, white_thr=245):
    h, _ = img.shape
    row_mean = img.mean(axis=1)
    valid = np.where(row_mean < white_thr)[0]
    if valid.size == 0:
        return img, (0, h)
    segs, start, prev = [], valid[0], valid[0]
    for y in valid[1:]:
        if y == prev + 1:
            prev = y
        else:
            segs.append((start, prev))
            start = prev = y
    segs.append((start, prev))
    y0, y1 = max(segs, key=lambda s: s[1] - s[0])
    return img[y0:y1 + 1, :], (y0, y1 + 1)

def apply_clahe(img_u8, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(img_u8)

def nlm_denoise(img, h_factor=1.15, patch_size=7, patch_distance=11):
    img_f = img.astype(np.float32) / 255.0
    sigma = float(np.mean(estimate_sigma(img_f, channel_axis=None)))
    den = denoise_nl_means(
        img_f, h=h_factor * sigma, fast_mode=True,
        patch_size=patch_size, patch_distance=patch_distance, channel_axis=None,
    )
    return (den * 255).astype(np.uint8)

# =============================================================
# 3. Beam removal — 연속된 bright row band 전체를 제거.
#    기존 argmax 한 줄 기반보다 beam 이 기울거나 두꺼울 때 안정적.
# =============================================================
def remove_central_beam_robust(img, k=3.0, max_half=None):
    H = img.shape[0]
    profile = img.mean(axis=1).astype(np.float32)
    thr = profile.mean() + k * profile.std()
    bright = profile >= thr
    if not bright.any():
        r0 = int(np.argmax(profile))
        half = max_half or 3
        return img.copy(), (max(0, r0 - half), min(H, r0 + half + 1))

    r_peak = int(np.argmax(profile))
    if not bright[r_peak]:
        idx = np.where(bright)[0]
        r_peak = int(idx[np.argmin(np.abs(idx - r_peak))])
    y0 = r_peak
    while y0 > 0 and bright[y0 - 1]:
        y0 -= 1
    y1 = r_peak
    while y1 < H - 1 and bright[y1 + 1]:
        y1 += 1
    y1 += 1
    if max_half is not None:
        mid = (y0 + y1) // 2
        y0 = max(y0, mid - max_half)
        y1 = min(y1, mid + max_half + 1)

    img2 = img.copy()
    img2[y0:y1, :] = int(np.median(img2))
    return img2, (y0, y1)

# =============================================================
# 4. Auto X-band — cornea/lens 가 있는 수평 범위를 밝은 열 분포로 자동 탐지.
# =============================================================
def detect_x_band(img, thr_k=1.0, margin_ratio=0.05,
                  fallback=(0.20, 0.60), min_ratio=0.05, max_ratio=0.80):
    H, W = img.shape
    bright = (img > (img.mean() + thr_k * img.std())).astype(np.float32)
    col = bright.sum(axis=0)
    if col.max() == 0:
        return int(W * fallback[0]), int(W * fallback[1])
    # smooth column profile
    win = max(5, int(W * 0.02)) | 1
    k = np.ones(win, dtype=np.float32) / win
    col_s = np.convolve(col, k, mode="same")
    thr = col_s.mean() + 0.5 * col_s.std()
    mask = col_s >= thr
    if not mask.any():
        return int(W * fallback[0]), int(W * fallback[1])
    xs = np.where(mask)[0]
    x0 = max(int(W * min_ratio), int(xs.min() - W * margin_ratio))
    x1 = min(int(W * max_ratio), int(xs.max() + W * margin_ratio))
    if x1 - x0 < int(W * 0.08):
        return int(W * fallback[0]), int(W * fallback[1])
    return x0, x1

# =============================================================
# 5. Cornea / Lens 검출 — CLAHE 된 이미지에서 threshold + 형태 정리.
# =============================================================
def get_cornea_lens_masks(img_norm, thr_k=1.5,
                           min_area_ratio=0.0008, dx_min_ratio=0.02,
                           close_r=3):
    H, W = img_norm.shape
    T = img_norm.mean() + thr_k * img_norm.std()
    binary = (img_norm > T).astype(np.uint8)
    if close_r > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1)),
        )
    lbl = label(binary.astype(bool))
    regs = regionprops(lbl)
    min_area = H * W * min_area_ratio
    big = [r for r in regs if r.area >= min_area]
    if len(big) < 2:
        raise RuntimeError(
            f"Only {len(big)} large bright blobs (need ≥2). "
            f"Try lowering thr_k or adjusting CLAHE."
        )
    big_sorted = sorted(big, key=lambda r: r.bbox[1])
    cornea_r = big_sorted[0]
    cornea_x0 = cornea_r.bbox[1]
    dx_min = int(W * dx_min_ratio)
    cand = [r for r in big if (r.bbox[1] - cornea_x0) >= dx_min]
    if cand:
        lens_r = max(cand, key=lambda r: r.area)
    elif len(big_sorted) >= 2:
        lens_r = big_sorted[1]
    else:
        raise RuntimeError("No lens candidate found.")
    return (lbl == cornea_r.label), (lbl == lens_r.label)

# =============================================================
# 6. Arc fitting — 반복적 outlier 제거 polynomial fit (RANSAC-lite).
#    row-wise max/min 에 섞인 noise 점을 자동으로 걸러서
#    안정적인 곡선을 복원.
# =============================================================
def fit_arc_polynomial(mask, side="posterior", degree=2,
                        n_iter=5, resid_k=2.5, beam_band=None):
    H, W = mask.shape
    xs_raw = np.full(H, np.nan, dtype=np.float32)
    for y in range(H):
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue
        xs_raw[y] = xs.max() if side == "posterior" else xs.min()

    if beam_band is not None:
        y0, y1 = beam_band
        xs_raw[max(0, y0):min(H, y1)] = np.nan

    valid = ~np.isnan(xs_raw)
    if valid.sum() < degree + 2:
        return xs_raw, None, float("inf")

    ys_v = np.where(valid)[0].astype(np.float32)
    xs_v = xs_raw[valid]
    keep = np.ones(ys_v.size, dtype=bool)
    coef = None
    final_std = float("inf")

    for _ in range(n_iter):
        if keep.sum() < degree + 2:
            break
        coef = np.polyfit(ys_v[keep], xs_v[keep], degree)
        pred = np.polyval(coef, ys_v)
        resid = xs_v - pred
        s = float(np.std(resid[keep]))
        final_std = s
        if s < 1e-6:
            break
        new_keep = np.abs(resid) <= resid_k * s
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if coef is None:
        return xs_raw, None, final_std

    ys_all = np.arange(H, dtype=np.float32)
    xs_fit = np.polyval(coef, ys_all).astype(np.float32)

    y_min = ys_v[keep].min() if keep.any() else ys_v.min()
    y_max = ys_v[keep].max() if keep.any() else ys_v.max()
    xs_fit[:int(y_min)] = np.nan
    xs_fit[int(y_max) + 1:] = np.nan
    xs_fit = np.clip(xs_fit, 0, W - 1)
    return xs_fit, coef, final_std

# =============================================================
# 7. AC polygon (original 방식 유지)
# =============================================================
def build_ac_mask_with_chords(img_shape, xs_cornea, xs_lens, min_width=5):
    xs_c = xs_cornea.astype(float).copy()
    xs_l = xs_lens.astype(float).copy()
    xs_c[(xs_c <= 0) | ~np.isfinite(xs_c)] = np.nan
    xs_l[(xs_l <= 0) | ~np.isfinite(xs_l)] = np.nan

    ys_c = np.where(~np.isnan(xs_c))[0]
    ys_l = np.where(~np.isnan(xs_l))[0]
    if ys_c.size < 2 or ys_l.size < 2:
        raise RuntimeError("Fitted arc too short.")
    overlap = np.intersect1d(ys_c, ys_l)
    if overlap.size > 0 and np.all((xs_l[overlap] - xs_c[overlap]) <= min_width):
        raise RuntimeError("AC width too narrow between cornea and lens arcs.")

    ys_c_s = np.sort(ys_c)
    ys_l_s = np.sort(ys_l)
    pts_c = np.stack([xs_c[ys_c_s], ys_c_s], axis=1).astype(np.int32)
    pts_l = np.stack([xs_l[ys_l_s], ys_l_s], axis=1).astype(np.int32)

    poly = pts_c.tolist() + [pts_l[-1].tolist()] + pts_l[::-1].tolist() + [pts_c[0].tolist()]
    polygon = np.array(poly, dtype=np.int32)
    mask = np.zeros(img_shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)

# =============================================================
# 8. Cell detection
#    (a) white_tophat(small disk) + Otsu  — 작은 밝은 blob 강조
#    (b) LoG blob detector — 절대 밝기 무관한 scale-space 검출
# =============================================================
def detect_cells_tophat(img_gray, mask_roi, disk_r,
                         T=None, area_min=1, area_max=30, circ_min=0.3):
    th = white_tophat(img_gray, disk(disk_r))
    if T is None:
        vals = th[mask_roi] if mask_roi is not None else th.ravel()
        vals = vals[vals > 0]
        T = float(threshold_otsu(vals)) if vals.size > 10 else 10.0
    binary = (th > T).astype(np.uint8)
    if mask_roi is not None:
        binary[~mask_roi] = 0
    lbl = label(binary.astype(bool))
    cells = []
    for r in regionprops(lbl):
        if r.area < area_min or r.area > area_max:
            continue
        p = r.perimeter
        circ = (4 * np.pi * r.area / (p ** 2)) if p > 0 else 1.0
        if circ < circ_min:
            continue
        cells.append(r)
    return cells, (binary * 255).astype(np.uint8), float(T)

def detect_cells_log(img_gray, mask_roi, min_sigma, max_sigma, threshold):
    img_f = img_gray.astype(np.float32) / 255.0
    if mask_roi is not None:
        img_f = img_f * mask_roi.astype(np.float32)
    blobs = blob_log(img_f, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=5, threshold=threshold)
    if mask_roi is not None and len(blobs) > 0:
        ys = blobs[:, 0].astype(int).clip(0, mask_roi.shape[0] - 1)
        xs = blobs[:, 1].astype(int).clip(0, mask_roi.shape[1] - 1)
        blobs = blobs[mask_roi[ys, xs]]
    return blobs

# =============================================================
# 9. Overlays
# =============================================================
def _gray_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def overlay_cells_roi(img_roi, cells, mask_roi=None):
    base = img_roi.copy()
    if mask_roi is not None:
        base[~mask_roi] = 0
    bgr = _gray_to_bgr(base)
    for r in cells:
        cy, cx = r.centroid
        cv2.circle(bgr, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_blobs_roi(img_roi, blobs, mask_roi=None):
    base = img_roi.copy()
    if mask_roi is not None:
        base[~mask_roi] = 0
    bgr = _gray_to_bgr(base)
    for y, x, s in blobs:
        cv2.circle(bgr, (int(x), int(y)), max(2, int(s * 1.414)), (0, 0, 255), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_cells_full(img_gray, cells, x0, y0, roi_shape):
    bgr = _gray_to_bgr(img_gray)
    for r in cells:
        cy, cx = r.centroid
        cv2.circle(bgr, (int(cx) + x0, int(cy) + y0), 3, (0, 0, 255), -1)
    h, w = roi_shape
    cv2.rectangle(bgr, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_blobs_full(img_gray, blobs, x0, y0, roi_shape):
    bgr = _gray_to_bgr(img_gray)
    for y, x, s in blobs:
        cv2.circle(bgr, (int(x) + x0, int(y) + y0),
                   max(2, int(s * 1.414)), (0, 0, 255), 1)
    h, w = roi_shape
    cv2.rectangle(bgr, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_ac_mask(img_gray, ac_mask):
    bgr = _gray_to_bgr(img_gray)
    ac_u8 = (ac_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(ac_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# =============================================================
# 10. Streamlit UI
# =============================================================
st.set_page_config(page_title="AC Cell Counter (Robust)", layout="wide")
st.title("IOLMaster AC Cell Quantification — Resolution/Noise Robust")

uploaded = st.file_uploader(
    "Upload IOLMaster B-scan (PNG/JPG/TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img_raw = to_gray_np(uploaded)
img_orig, (crop_y0, crop_y1) = autocrop_vertical_white(img_raw)
H, W = img_orig.shape
P = get_norm_params(H, W)

st.sidebar.header("0. Calibration")
pixel_um = st.sidebar.number_input("µm / pixel (0 = unknown)", 0.0, 100.0, 0.0, 0.5)

st.sidebar.header("1. Preprocessing")
use_clahe = st.sidebar.checkbox("Apply CLAHE (contrast normalize)", True)
clahe_clip = st.sidebar.slider("CLAHE clip limit", 0.5, 5.0, 2.0, 0.1)
h_factor = st.sidebar.slider("NLM h factor", 0.5, 3.0, 1.15, 0.05)

st.sidebar.header("2. Beam removal")
beam_k = st.sidebar.slider("Beam k (mean + k·σ)", 1.0, 6.0, float(P["beam_k"]), 0.1)
beam_cap = st.sidebar.slider("Beam half-width cap (rows)", 3, 80, int(P["beam_half_max"]))

st.sidebar.header("3. Cornea/Lens detection")
auto_xband = st.sidebar.checkbox("Auto-detect X band", True)
x0_manual = st.sidebar.slider("Manual X0", 0, max(10, W - 10), int(W * 0.25))
x1_manual = st.sidebar.slider("Manual X1", 10, W, int(W * 0.55))
thr_k = st.sidebar.slider("Threshold k (on CLAHE/NLM)", 0.2, 4.0,
                           float(P["cornea_thr_k"]), 0.1)

st.sidebar.header("4. Arc fitting")
poly_deg = st.sidebar.slider("Polynomial degree", 2, 4, int(P["poly_degree"]))
resid_k = st.sidebar.slider("Outlier reject k·σ", 1.0, 4.0,
                             float(P["poly_resid_k"]), 0.1)
min_width = st.sidebar.slider("Min AC width per row (px)", 1, 40, 5)

st.sidebar.header("5. Cell detection")
detector = st.sidebar.radio("Method", ["Top-hat + Otsu", "LoG blob"])
cell_disk_r = st.sidebar.slider("Cell top-hat disk radius (px)",
                                  1, 20, int(P["cell_disk"]))
manual_T = st.sidebar.checkbox("Manual top-hat threshold", False)
T_manual = st.sidebar.slider("Top-hat T", 0, 255, 15) if manual_T else None
area_min = st.sidebar.slider("Min area (px²)", 1, 100, 2)
area_max = st.sidebar.slider("Max area (px²)", 5, 300, 30)
circ_min = st.sidebar.slider("Min circularity", 0.0, 1.0, 0.3, 0.05)
log_min_sigma = st.sidebar.slider("LoG min σ", 0.5, 5.0, 0.8, 0.1)
log_max_sigma = st.sidebar.slider("LoG max σ", 1.0, 10.0, 3.0, 0.1)
log_thr = st.sidebar.slider("LoG threshold", 0.001, 0.3, 0.02, 0.001)

step = st.radio(
    "View stage:",
    ["Original", "Preprocessed", "Beam removed", "X band",
     "Cornea/Lens mask", "Arc fit", "AC ROI", "Cells"],
    horizontal=True,
)

# -------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------
img_nlm = nlm_denoise(img_orig, h_factor=h_factor)
img_beam, beam_band = remove_central_beam_robust(img_nlm, k=beam_k, max_half=beam_cap)
img_seg = apply_clahe(img_beam, clip=clahe_clip) if use_clahe else img_beam

if auto_xband:
    x0, x1 = detect_x_band(
        img_seg, thr_k=thr_k * 0.7,
        margin_ratio=P["x_band_margin"], fallback=P["x_band_fallback"],
    )
else:
    x0 = min(x0_manual, max(0, x1_manual - 10))
    x1 = max(x1_manual, x0_manual + 10)

status, error_msg = [], None
cells, blobs = [], np.zeros((0, 3))
binary = ac_mask_full = img_roi = mask_roi = None
cornea_mask_full = lens_mask_full = None
xs_cornea_fit = xs_lens_fit = None
overlay_roi_img = overlay_full_img = None
xx0 = yy0 = None

try:
    img_band_seg = img_seg[:, x0:x1]
    cornea_mask, lens_mask = get_cornea_lens_masks(
        img_band_seg, thr_k=thr_k,
        min_area_ratio=P["min_area_ratio"],
        dx_min_ratio=P["dx_min_ratio"],
    )

    cornea_mask_full = np.zeros_like(img_orig, dtype=bool)
    cornea_mask_full[:, x0:x1] = cornea_mask
    lens_mask_full = np.zeros_like(img_orig, dtype=bool)
    lens_mask_full[:, x0:x1] = lens_mask

    bb = (max(0, beam_band[0] - P["beam_pad"]),
          min(H, beam_band[1] + P["beam_pad"]))

    xs_cornea_fit, coef_c, std_c = fit_arc_polynomial(
        cornea_mask, side="posterior", degree=poly_deg,
        n_iter=P["poly_iter"], resid_k=resid_k, beam_band=bb,
    )
    xs_lens_fit, coef_l, std_l = fit_arc_polynomial(
        lens_mask, side="anterior", degree=poly_deg,
        n_iter=P["poly_iter"], resid_k=resid_k, beam_band=bb,
    )
    if coef_c is None or coef_l is None:
        raise RuntimeError("Arc fit failed (too few valid points after outlier rejection).")
    status.append(f"Cornea fit residual σ = {std_c:.2f} px")
    status.append(f"Lens fit residual σ   = {std_l:.2f} px")

    ac_mask_band = build_ac_mask_with_chords(
        img_shape=img_band_seg.shape,
        xs_cornea=xs_cornea_fit, xs_lens=xs_lens_fit,
        min_width=min_width,
    )
    ac_mask_full = np.zeros_like(img_orig, dtype=bool)
    ac_mask_full[:, x0:x1] = ac_mask_band

    ys, xs = np.where(ac_mask_full)
    yy0, yy1 = int(ys.min()), int(ys.max() + 1)
    xx0, xx1 = int(xs.min()), int(xs.max() + 1)
    img_roi = img_beam[yy0:yy1, xx0:xx1]
    mask_roi = ac_mask_full[yy0:yy1, xx0:xx1]

    if detector == "Top-hat + Otsu":
        cells, binary, T_used = detect_cells_tophat(
            img_roi, mask_roi=mask_roi, disk_r=cell_disk_r,
            T=T_manual, area_min=area_min, area_max=area_max, circ_min=circ_min,
        )
        status.append(f"Cell top-hat T = {T_used:.1f}")
        overlay_roi_img = overlay_cells_roi(img_roi, cells, mask_roi=mask_roi)
        overlay_full_img = overlay_cells_full(img_orig, cells, xx0, yy0, img_roi.shape)
    else:
        blobs = detect_cells_log(
            img_roi, mask_roi=mask_roi,
            min_sigma=log_min_sigma, max_sigma=log_max_sigma, threshold=log_thr,
        )
        overlay_roi_img = overlay_blobs_roi(img_roi, blobs, mask_roi=mask_roi)
        overlay_full_img = overlay_blobs_full(img_orig, blobs, xx0, yy0, img_roi.shape)

except Exception as e:
    error_msg = str(e)

# -------------------------------------------------------------
# Header metrics
# -------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Image (H × W)", f"{H} × {W}")
m1.metric("X band", f"{x0} – {x1}")
m2.metric("Beam rows", f"{beam_band[0]} – {beam_band[1]}")
n_count = len(cells) if detector == "Top-hat + Otsu" else int(len(blobs))
m2.metric("Detected cells", n_count)
if pixel_um > 0 and ac_mask_full is not None:
    area_mm2 = float(ac_mask_full.sum()) * (pixel_um / 1000.0) ** 2
    m3.metric("AC area (mm²)", f"{area_mm2:.3f}")
    m3.metric("Density (cells/mm²)", f"{n_count / max(area_mm2, 1e-6):.1f}")

for s in status:
    st.caption(s)
if error_msg:
    st.warning(f"Pipeline error: {error_msg}")

# -------------------------------------------------------------
# Stage views
# -------------------------------------------------------------
if step == "Original":
    st.image(img_orig, caption="Vertically autocropped original", clamp=True)

elif step == "Preprocessed":
    c1, c2 = st.columns(2)
    c1.image(img_nlm, caption="NLM denoised", clamp=True)
    c2.image(img_seg, caption="For segmentation (CLAHE on beam-removed)", clamp=True)

elif step == "Beam removed":
    st.image(img_beam, caption=f"Beam rows {beam_band[0]}–{beam_band[1]} filled", clamp=True)

elif step == "X band":
    vis = _gray_to_bgr(img_seg)
    cv2.line(vis, (x0, 0), (x0, H), (0, 255, 0), 1)
    cv2.line(vis, (x1 - 1, 0), (x1 - 1, H), (0, 255, 0), 1)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
             caption=f"X band [{x0}, {x1}] ({'auto' if auto_xband else 'manual'})")

elif step == "Cornea/Lens mask":
    if cornea_mask_full is not None and lens_mask_full is not None:
        vis = _gray_to_bgr(img_seg)
        vis[cornea_mask_full] = (0, 255, 0)
        vis[lens_mask_full] = (255, 0, 0)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption="Green = cornea blob, Blue = lens blob")
    else:
        st.error(f"Mask unavailable. {error_msg or ''}")

elif step == "Arc fit":
    if xs_cornea_fit is not None and xs_lens_fit is not None:
        vis = _gray_to_bgr(img_seg)
        for y in range(H):
            xc = xs_cornea_fit[y]
            xl = xs_lens_fit[y]
            if np.isfinite(xc):
                cx = int(xc) + x0
                if 0 <= cx < W:
                    vis[y, cx] = (0, 255, 255)
            if np.isfinite(xl):
                cx = int(xl) + x0
                if 0 <= cx < W:
                    vis[y, cx] = (255, 0, 255)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption="Yellow = cornea posterior fit, Magenta = lens anterior fit")
    else:
        st.error(f"Arc fit unavailable. {error_msg or ''}")

elif step == "AC ROI":
    if ac_mask_full is not None:
        st.image(overlay_ac_mask(img_orig, ac_mask_full),
                 caption="AC ROI polygon on original image", clamp=True)
        if img_roi is not None:
            st.image(img_roi, caption="AC ROI crop (beam-removed)", clamp=True)
    else:
        st.error(f"AC ROI unavailable. {error_msg or ''}")

elif step == "Cells":
    if overlay_roi_img is not None:
        st.image(overlay_roi_img, caption=f"AC ROI + detected cells (n={n_count})",
                 clamp=True)
        st.image(overlay_full_img, caption="Full image + detected cells", clamp=True)
        if binary is not None:
            st.image(binary, caption="Top-hat binary inside AC ROI", clamp=True)
    else:
        st.error(f"Cell overlay unavailable. {error_msg or ''}")
