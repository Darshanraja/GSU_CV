"""
CSc 8830 - Computer Vision
SAM2 baseline mask  (COMPARISON ONLY - not part of the classical pipeline)

Usage:
    python sam2_mask.py --image rgb_original.png --prefix rgb
    python sam2_mask.py --image thermal_original.png --prefix thermal
    python sam2_mask.py --image me.jpg --box 120 40 640 900      # skip the window (x1 y1 x2 y2)

Workflow:
    1. Drag a rectangle around the person (use the same box as for GrabCut), press ENTER
    2. SAM2 predicts the person mask from that box
    3. Saves  <prefix>_sam2_mask.png     (white person on black)
              <prefix>_sam2_overlay.png  (boundary drawn on the image)
    4. Upload <prefix>_sam2_mask.png in the website's Comparison tab

Setup (Python >= 3.10, once):
    python3 -m venv sam2env && source sam2env/bin/activate
    pip install torch torchvision opencv-python huggingface_hub
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    pip install -e .                       # if it complains about CUDA/nvcc:
                                           #   SAM2_BUILD_CUDA=0 pip install -e .
    cd ..  &&  python sam2_mask.py --image rgb_original.png --prefix rgb

The first run downloads the model weights from Hugging Face automatically.
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for missing ops

import cv2
import numpy as np
import torch


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_predictor(model_id, device):
    # build_sam2() defaults to device="cuda", which fails on a Mac, so pass the device explicitly.
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2_hf(model_id, device=device)
    return SAM2ImagePredictor(model)


def segment_with_box(predictor, image_bgr, box, device):
    """box = (x1, y1, x2, y2) in original image pixels. Returns a uint8 mask (0 / 255)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    use_amp = device == "cuda"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,          # recommended for a single box prompt
        )
    mask = np.asarray(masks[0]) > 0
    print("SAM2 confidence score: %.3f" % float(np.asarray(scores).reshape(-1)[0]))
    return (mask.astype(np.uint8)) * 255


# ------------------------------------------------------------------
# Box selection
# ------------------------------------------------------------------
def select_box(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    display = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else image.copy()

    print("Drag a rectangle around the person, then press ENTER (C = cancel).")
    x, y, bw, bh = cv2.selectROI("Draw box around the person", display,
                                 showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    if bw == 0 or bh == 0:
        return None
    return (int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale))


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="the SAME image you segmented (e.g. rgb_original.png)")
    ap.add_argument("--prefix", default=None, help="output prefix (default: image file name)")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="skip the window and use this box")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large",
                    help="e.g. facebook/sam2.1-hiera-tiny is much faster on a laptop")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print("Could not read", args.image)
        return
    prefix = args.prefix or os.path.splitext(os.path.basename(args.image))[0]

    box = tuple(args.box) if args.box else select_box(image)
    if box is None:
        print("No box selected.")
        return

    device = pick_device(args.device)
    print("Loading", args.model, "on", device, "(first run downloads the weights)...")
    predictor = load_predictor(args.model, device)

    mask = segment_with_box(predictor, image, box, device)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)      # red, so it differs from the green classical result

    mask_path = f"{prefix}_sam2_mask.png"
    overlay_path = f"{prefix}_sam2_overlay.png"
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    print("\n=== SAM2 RESULT ===")
    print("Mask area:", int(np.count_nonzero(mask)), "px")
    print("Saved:", mask_path, overlay_path)

    cv2.imshow("SAM2 mask", mask)
    cv2.imshow("SAM2 boundary", overlay)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()