"""
CSc 8830 - Computer Vision
Problem 1: RGB human boundary detection (classical CV only, no ML/DL)

Usage:
    python rgb_human_boundary.py                 # iPhone Continuity Camera
    python rgb_human_boundary.py --image me.jpg  # existing image (handy for SAM2 comparison)

Workflow:
    1. (camera) SPACE to capture
    2. Drag a rectangle around the person, ENTER to confirm
       -> the script AUTO-SEEDS GrabCut from colour statistics of the box
    3. Refine:  LEFT drag = human, RIGHT drag = background
                G = re-run GrabCut, R = reset to auto-seed, S = save, ESC = cancel
"""

import argparse
import cv2
import numpy as np

CAMERA_INDEX = 1
OUTPUT_ORIGINAL = "rgb_original.png"
OUTPUT_MASK = "rgb_human_mask.png"
OUTPUT_SEGMENTED = "rgb_segmented_human.png"
OUTPUT_BOUNDARY = "rgb_human_boundary.png"


# ------------------------------------------------------------------
# Capture
# ------------------------------------------------------------------
def capture_iphone_image():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open iPhone Continuity Camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break
        disp = frame.copy()
        cv2.putText(disp, "SPACE: Capture   ESC: Exit", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("iPhone Continuity Camera", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured = frame.copy()
            break
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured


def resize_for_display(image, max_width=800, max_height=800):
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


# ------------------------------------------------------------------
# Step 1: rectangle
# ------------------------------------------------------------------
def select_human_rectangle(image):
    display, scale = resize_for_display(image, 1200, 800)
    state = {"drawing": False, "p1": None, "p2": None, "rect": None}
    win = "Step 1 - Select Human"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(drawing=True, p1=(x, y), p2=(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["p2"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["p2"] = (x, y)
            (ax, ay), (bx, by) = state["p1"], state["p2"]
            x1, x2, y1, y2 = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
            if x2 - x1 > 5 and y2 - y1 > 5:
                state["rect"] = (int(x1 / scale), int(y1 / scale),
                                 int((x2 - x1) / scale), int((y2 - y1) / scale))

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("STEP 1: drag a TIGHT rectangle around the person. ENTER=confirm R=reset ESC=cancel")

    while True:
        frame = display.copy()
        if state["p1"] and state["p2"]:
            cv2.rectangle(frame, state["p1"], state["p2"], (0, 255, 0), 3)
        cv2.putText(frame, "Drag around person | ENTER confirm | R reset", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and state["rect"] is not None:
            break
        elif key in (ord("r"), ord("R")):
            state.update(p1=None, p2=None, rect=None)
        elif key == 27:
            state["rect"] = None
            break
    cv2.destroyWindow(win)
    return state["rect"]


# ------------------------------------------------------------------
# Auto-seeding (the fix)
# ------------------------------------------------------------------
def build_initial_mask(image, rect, edge_frac=0.10, thresh=9.0):
    """
    Seed GrabCut with more than just "everything in the box is foreground".

    1. Outside the box                       -> sure background
    2. Learn the background colour (Lab mean/covariance) from the two TOP
       CORNERS and the upper side edges of the box (top-centre is skipped
       because that is where hair/head usually is).
    3. Pixels in the box close to that colour (Mahalanobis distance)
                                             -> probable background
       everything else                       -> probable foreground
    4. Background-coloured pixels touching the top corners / upper sides
                                             -> sure background
    5. Torso prior: lower-middle of the box  -> never "background-coloured"
       (a light shirt is often close to a light wall)
    6. Small ellipse in the middle of the box -> sure foreground
    """
    h, w = image.shape[:2]
    x, y, rw, rh = rect
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)

    lab = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    roi = lab[y:y + rh, x:x + rw]

    bw = max(5, int(rw * edge_frac))
    bh = max(5, int(rh * edge_frac))
    cw = int(rw * 0.25)
    side_h = int(rh * 0.5)

    samples = np.concatenate([
        roi[:bh * 2, :cw].reshape(-1, 3),
        roi[:bh * 2, -cw:].reshape(-1, 3),
        roi[:side_h, :bw].reshape(-1, 3),
        roi[:side_h, -bw:].reshape(-1, 3),
    ])
    mean = samples.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(samples.T) + 4.0 * np.eye(3))
    d = roi - mean
    maha2 = np.einsum("ijk,kl,ijl->ij", d, inv_cov, d)
    likely_bg = maha2 < thresh

    torso = np.zeros((rh, rw), bool)
    torso[int(rh * 0.65):, int(rw * 0.15):int(rw * 0.85)] = True
    likely_bg &= ~torso

    sub = np.where(likely_bg, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype(np.uint8)

    band = np.zeros((rh, rw), bool)
    band[:bh, :cw] = True
    band[:bh, -cw:] = True
    band[:side_h, :bw] = True
    band[:side_h, -bw:] = True
    sub[band & likely_bg] = cv2.GC_BGD

    core = np.zeros((rh, rw), np.uint8)
    cv2.ellipse(core, (rw // 2, int(rh * 0.6)),
                (max(3, int(rw * 0.10)), max(3, int(rh * 0.18))),
                0, 0, 360, 1, -1)
    sub[core == 1] = cv2.GC_FGD

    mask[y:y + rh, x:x + rw] = sub
    return mask


def to_binary(mask):
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


# ------------------------------------------------------------------
# Step 2: interactive GrabCut
# ------------------------------------------------------------------
def interactive_grabcut(image, rectangle):
    seed_mask = build_initial_mask(image, rectangle)
    mask = seed_mask.copy()
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

    # side-by-side window must fit on screen, so each panel is <= 800 px wide
    display, scale = resize_for_display(image, 800, 800)
    paint = display.copy()
    panel_w = display.shape[1]
    st = {"drawing": False, "label": None}
    brush = 8
    win = "Step 2 - Refine Segmentation"

    def paint_at(dx, dy, label):
        if dx >= panel_w:            # ignore clicks on the preview panel
            return
        cv2.circle(mask, (int(dx / scale), int(dy / scale)),
                   max(2, int(brush / scale)), int(label), -1)
        color = (0, 255, 0) if label == cv2.GC_FGD else (0, 0, 255)
        cv2.circle(paint, (dx, dy), brush, color, -1)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.update(drawing=True, label=cv2.GC_FGD); paint_at(x, y, st["label"])
        elif event == cv2.EVENT_RBUTTONDOWN:
            st.update(drawing=True, label=cv2.GC_BGD); paint_at(x, y, st["label"])
        elif event == cv2.EVENT_MOUSEMOVE and st["drawing"]:
            paint_at(x, y, st["label"])
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            st.update(drawing=False, label=None)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)   # AUTOSIZE keeps mouse coords 1:1
    cv2.setMouseCallback(win, on_mouse)
    print("STEP 2: LEFT=human  RIGHT=background  G=refine  R=reset  S=save  ESC=cancel")

    while True:
        b = to_binary(mask)
        preview = cv2.bitwise_and(image, image, mask=b)
        preview = cv2.resize(preview, (display.shape[1], display.shape[0]),
                             interpolation=cv2.INTER_AREA)
        # draw current boundary on the left panel
        left = paint.copy()
        cnts, _ = cv2.findContours(cv2.resize(b, (display.shape[1], display.shape[0]),
                                              interpolation=cv2.INTER_NEAREST),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(left, cnts, -1, (255, 255, 0), 1)
        combined = np.hstack((left, preview))
        cv2.putText(combined, "LEFT=Human RIGHT=Background G=Refine R=Reset S=Save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(win, combined)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("g"), ord("G")):
            cv2.grabCut(image, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            print("GrabCut refined.")
        elif key in (ord("r"), ord("R")):
            mask = seed_mask.copy()
            bgd[:] = 0; fgd[:] = 0
            cv2.grabCut(image, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            paint = display.copy()
            print("Reset.")
        elif key in (ord("s"), ord("S")):
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return to_binary(mask)


# ------------------------------------------------------------------
# Post-processing / boundary
# ------------------------------------------------------------------
def keep_largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def clean_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # fill interior holes (e.g. gaps between arm and torso marked wrongly)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def find_boundary(image, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    contour = max(cnts, key=cv2.contourArea)
    out = image.copy()
    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
    return out, contour


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="use this image instead of the camera")
    args = ap.parse_args()

    image = cv2.imread(args.image) if args.image else capture_iphone_image()
    if image is None:
        print("No image.")
        return
    cv2.imwrite(OUTPUT_ORIGINAL, image)

    rect = select_human_rectangle(image)
    if rect is None:
        print("No rectangle selected.")
        return

    mask = interactive_grabcut(image, rect)
    if mask is None:
        return

    mask = clean_mask(keep_largest_component(mask))
    boundary_image, contour = find_boundary(image, mask)
    if contour is None:
        print("No contour found.")
        return
    segmented = cv2.bitwise_and(image, image, mask=mask)

    print("\n=== RGB HUMAN SEGMENTATION RESULT ===")
    print("Contour area   :", cv2.contourArea(contour), "px")
    print("Boundary length:", cv2.arcLength(contour, True), "px")
    print("Boundary points:", len(contour))

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