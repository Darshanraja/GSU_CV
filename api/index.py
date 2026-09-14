import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class CalibrationRequest(BaseModel):
    images: List[str]
    checkerboard_cols: int = 9
    checkerboard_rows: int = 6
    square_size_cm: float = 2.5

@app.get("/api")
def health():
    return {"ok": True, "service": "GSU CV Module 2 calibration API"}

def decode_data_url(data_url: str):
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")
    return image

@app.post("/api/calibrate")
def calibrate_camera(payload: CalibrationRequest):
    if len(payload.images) < 6:
        raise HTTPException(status_code=400, detail="At least 6 images are required.")

    checkerboard = (payload.checkerboard_cols, payload.checkerboard_rows)
    square_size_m = payload.square_size_cm / 100.0

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    object_points = []
    image_points = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    for encoded in payload.images:
        try:
            frame = decode_data_url(encoded)
        except Exception:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        if (gray.shape[1], gray.shape[0]) != image_size:
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            try:
                found_sb, corners_sb = cv2.findChessboardCornersSB(
                    gray,
                    checkerboard,
                    flags=(
                        cv2.CALIB_CB_NORMALIZE_IMAGE
                        + cv2.CALIB_CB_EXHAUSTIVE
                        + cv2.CALIB_CB_ACCURACY
                    )
                )
                if found_sb:
                    found = True
                    corners = corners_sb
            except cv2.error:
                pass

        if not found:
            continue

        try:
            refined = cv2.cornerSubPix(
                gray,
                corners.astype(np.float32),
                (11, 11),
                (-1, -1),
                criteria
            )
        except cv2.error:
            refined = corners.astype(np.float32)

        object_points.append(objp.copy())
        image_points.append(refined)

    valid_images = len(object_points)

    if valid_images < 6:
        raise HTTPException(
            status_code=400,
            detail=f"Only {valid_images} valid checkerboard images were detected. At least 6 are required."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    return {
        "success": True,
        "total_images": len(payload.images),
        "valid_images": valid_images,
        "rms": float(rms),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "distortion": [float(x) for x in dist_coeffs.reshape(-1)]
    }
