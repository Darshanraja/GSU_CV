"""
CSc 8830 - Computer Vision
Problem 2: Thermal human boundary detection (classical CV only, no ML/DL)

Usage:
    python thermal_human_boundary.py --image thermal.jpg
    python thermal_human_boundary.py --image thermal.jpg --auto      # no GUI, fully automatic
    python thermal_human_boundary.py --image thermal.jpg --invert    # black-hot images

Idea:
    A human is warmer than most backgrounds, so in a "white-hot" thermal image
    the person is BRIGHT. Pipeline:
        blur -> contrast stretch -> threshold (starts at Otsu) -> morphology
        -> fill holes -> choose the human component(s) -> optional GrabCut
        refinement of the edge -> contour

Interactive controls:
    Trackbar "Threshold" = temperature cut-off (starts at Otsu's value)
    Trackbar "Close"     = closing size (joins cooler clothing to warm face/hands)
    LEFT click on a blob = select / deselect it as the human
                           (nothing selected -> largest blob is used)
    C = clear selection    I = invert (black-hot)    G = toggle GrabCut edge refinement
    S = save               ESC = cancel
"""

import argparse
import cv2
import numpy as np

OUTPUT_ORIGINAL = "thermal_original.png"
OUTPUT_MASK = "thermal_human_mask.png"
OUTPUT_SEGMENTED = "thermal_segmented_human.png"
OUTPUT_BOUNDARY = "thermal_human_boundary.png"


# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------
def preprocess(bgr):
    """Grayscale + light blur + contrast stretch (thermal images are often low contrast)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def resize_for_display(image, max_width=700, max_height=800):
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


# ------------------------------------------------------------------
# Segmentation steps
# ------------------------------------------------------------------
def threshold_mask(smooth, thr, invert):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, m = cv2.threshold(smooth, thr, 255, mode)
    return m


def cleanup(mask, close_size):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)            # remove speckle
    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_size + 1, 2 * close_size + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)        # bridge cool clothing gaps
    return mask


def fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def select_components(mask, points):
    """Keep the blobs under the clicked points; if none clicked, keep the largest blob."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask), labels
    keep = {int(labels[py, px]) for (px, py) in points if labels[py, px] > 0}
    if not keep:
        keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    out = np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)
    return out, labels


def grabcut_refine(bgr, mask, band=None):
    """
    Sharpen the edge of a thresholded mask with GrabCut.
    Deep inside the mask = sure human, far outside = sure background,
    a thin band around the edge = undecided, so GrabCut only decides the boundary.
    """
    h, w = mask.shape
    band = band or max(3, int(0.01 * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_fg = cv2.erode(mask, k)
    maybe = cv2.dilate(mask, k)

    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD

    fg_n = np.count_nonzero((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    bg_n = np.count_nonzero((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD))
    if fg_n < 100 or bg_n < 100:
        return mask

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def build_mask(bgr, smooth, thr, close_size, invert, points, refine):
    m = threshold_mask(smooth, thr, invert)
    m = fill_holes(cleanup(m, close_size))
    selected, labels = select_components(m, points)
    if refine:
        selected = fill_holes(grabcut_refine(bgr, selected))
    return selected, labels


def otsu_threshold(smooth):
    t, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


# ------------------------------------------------------------------
# Interactive UI
# ------------------------------------------------------------------
def interactive_segmentation(bgr, smooth, invert=False):
    display, scale = resize_for_display(bgr)
    panel_w = display.shape[1]
    win = "Thermal Human Segmentation"

    state = {"points": [], "labels": None, "ver": 0}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or x >= panel_w or state["labels"] is None:
            return
        ox, oy = int(x / scale), int(y / scale)
        h, w = state["labels"].shape
        if not (0 <= ox < w and 0 <= oy < h):
            return
        lab = state["labels"][oy, ox]
        if lab == 0:
            return
        same = [p for p in state["points"] if state["labels"][p[1], p[0]] == lab]
        if same:                                   # clicking a selected blob deselects it
            state["points"] = [p for p in state["points"] if p not in same]
        else:
            state["points"].append((ox, oy))
        state["ver"] += 1

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Threshold", win, otsu_threshold(smooth), 255, lambda v: None)
    cv2.createTrackbar("Close", win, 5, 25, lambda v: None)
    cv2.setMouseCallback(win, on_mouse)

    print("Trackbars: Threshold, Close | click blob = select human")
    print("C=clear  I=invert  G=GrabCut refine  S=save  ESC=cancel")

    refine = False
    last_params = None
    mask = np.zeros(smooth.shape, np.uint8)
    combined = None

    while True:
        thr = cv2.getTrackbarPos("Threshold", win)
        close = cv2.getTrackbarPos("Close", win)
        params = (thr, close, invert, refine, state["ver"])

        if params != last_params:
            last_params = params
            mask, labels = build_mask(bgr, smooth, thr, close, invert,
                                      state["points"], refine)
            state["labels"] = labels

            # left panel: candidate blobs (yellow) + selected human boundary (green)
            left = display.copy()
            cand = fill_holes(cleanup(threshold_mask(smooth, thr, invert), close))
            for src, color, t in ((cand, (0, 255, 255), 1), (mask, (0, 255, 0), 2)):
                small = cv2.resize(src, (panel_w, display.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(left, cnts, -1, color, t)

            seg = cv2.bitwise_and(bgr, bgr, mask=mask)
            seg = cv2.resize(seg, (panel_w, display.shape[0]), interpolation=cv2.INTER_AREA)
            combined = np.hstack((left, seg))
            cv2.putText(combined,
                        f"thr={thr} close={close} invert={invert} grabcut={refine} | "
                        "click=select C=clear I=invert G=refine S=save",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(win, combined)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("i"), ord("I")):
            invert = not invert
        elif key in (ord("g"), ord("G")):
            refine = not refine
        elif key in (ord("c"), ord("C")):
            state["points"] = []
            state["ver"] += 1
        elif key in (ord("s"), ord("S")):
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return mask


# ------------------------------------------------------------------
# Boundary + measurements
# ------------------------------------------------------------------
def find_boundary(image, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, []
    out = image.copy()
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out, cnts


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="thermal image file")
    ap.add_argument("--auto", action="store_true", help="no GUI: Otsu + largest blob + GrabCut")
    ap.add_argument("--invert", action="store_true", help="black-hot image (human is dark)")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        print("Could not read", args.image)
        return
    cv2.imwrite(OUTPUT_ORIGINAL, bgr)
    smooth = preprocess(bgr)

    if args.auto:
        mask, _ = build_mask(bgr, smooth, otsu_threshold(smooth), 5,
                             args.invert, [], refine=True)
    else:
        mask = interactive_segmentation(bgr, smooth, args.invert)
        if mask is None:
            return

    boundary_image, contours = find_boundary(bgr, mask)
    if not contours:
        print("No contour found.")
        return
    segmented = cv2.bitwise_and(bgr, bgr, mask=mask)

    print("\n=== THERMAL HUMAN SEGMENTATION RESULT ===")
    for i, c in enumerate(contours, 1):
        print(f"Human {i}: area={cv2.contourArea(c):.0f} px  "
              f"boundary length={cv2.arcLength(c, True):.0f} px  points={len(c)}")

    cv2.imwrite(OUTPUT_MASK, mask)
    cv2.imwrite(OUTPUT_SEGMENTED, segmented)
    cv2.imwrite(OUTPUT_BOUNDARY, boundary_image)
    print("Saved:", OUTPUT_ORIGINAL, OUTPUT_MASK, OUTPUT_SEGMENTED, OUTPUT_BOUNDARY)

    cv2.imshow("Final Human Mask", mask)
    cv2.imshow("Final Segmented Human", segmented)
    cv2.imshow("Final Human Boundary", boundary_image)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()