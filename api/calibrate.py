from http.server import BaseHTTPRequestHandler
import json
import base64
import cv2
import numpy as np

def send_json(h, status, payload):
    body = json.dumps(payload).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)

def decode_data_url(data_url):
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

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            images = payload.get("images", [])
            checkerboard_cols = int(payload.get("checkerboard_cols", 9))
            checkerboard_rows = int(payload.get("checkerboard_rows", 6))
            square_size_cm = float(payload.get("square_size_cm", 2.5))

            if len(images) < 6:
                return send_json(self, 400, {
                    "detail": "At least 6 checkerboard images are required."
                })

            checkerboard = (checkerboard_cols, checkerboard_rows)
            square_size_m = square_size_cm / 100.0

            objp = np.zeros(
                (checkerboard[0] * checkerboard[1], 3),
                np.float32
            )

            objp[:, :2] = (
                np.mgrid[
                    0:checkerboard[0],
                    0:checkerboard[1]
                ]
                .T
                .reshape(-1, 2)
            )

            objp *= square_size_m

            object_points = []
            image_points = []
            image_size = None

            criteria = (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            for encoded in images:
                try:
                    frame = decode_data_url(encoded)
                except Exception:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                size = (gray.shape[1], gray.shape[0])

                if image_size is None:
                    image_size = size

                if size != image_size:
                    continue

                found, corners = cv2.findChessboardCorners(
                    gray,
                    checkerboard,
                    flags=(
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
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
                return send_json(self, 400, {
                    "detail": (
                        f"Only {valid_images} valid checkerboard images "
                        "were detected. At least 6 are required."
                    )
                })

            rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None
            )

            return send_json(self, 200, {
                "success": True,
                "total_images": len(images),
                "valid_images": valid_images,
                "rms": float(rms),
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "image_width": int(image_size[0]),
                "image_height": int(image_size[1]),
                "distortion": [
                    float(x)
                    for x in dist_coeffs.reshape(-1)
                ]
            })

        except Exception as exc:
            return send_json(self, 500, {
                "detail": f"Calibration server error: {str(exc)}"
            })
