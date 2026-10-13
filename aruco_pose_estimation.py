#!/usr/bin/env python3

import argparse
import math
import sys
import time

import cv2
import numpy as np

aruco = cv2.aruco

# The *_1000 dictionaries contain the first 50/100/250 markers of the smaller
# ones, so a single dictionary per grid size covers every ID you can generate.
DICTIONARIES = {
    "4x4": aruco.DICT_4X4_1000,
    "5x5": aruco.DICT_5X5_1000,
    "6x6": aruco.DICT_6X6_1000,
}

# BGR colours used to tell the marker families apart on screen
COLORS = {
    "4x4": (0, 255, 0),
    "5x5": (255, 160, 0),
    "6x6": (0, 140, 255),
}

# Marker frame from solvePnP has y up and z pointing out of the marker towards
# the camera. This flips it to the camera convention (x right, y down, z into
# the marker) so that a marker squarely facing the camera has roll=pitch=yaw=0.
FLIP = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------- #
# Detection helpers (work with both the old and new OpenCV ArUco APIs)
# --------------------------------------------------------------------------- #
def build_detector(dictionary_id):
    """Return a function gray_image -> (corners, ids, rejected)."""
    dictionary = aruco.getPredefinedDictionary(dictionary_id)

    if hasattr(aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)

    params = aruco.DetectorParameters_create()  # OpenCV < 4.7
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def load_camera_matrix(calib_path, frame_shape):
    """Load calibration from .npz, or build a rough approximation."""
    if calib_path:
        data = np.load(calib_path)
        camera_matrix = data["camera_matrix"].astype(np.float64)
        dist_coeffs = data["dist_coeffs"].astype(np.float64)
        return camera_matrix, dist_coeffs, True

    h, w = frame_shape[:2]
    focal = float(w)  # rough guess: f ~ image width (about 53 deg horizontal FOV)
    camera_matrix = np.array(
        [[focal, 0.0, w / 2.0],
         [0.0, focal, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return camera_matrix, np.zeros((5, 1), dtype=np.float64), False


# --------------------------------------------------------------------------- #
# Pose helpers
# --------------------------------------------------------------------------- #
def estimate_marker_pose(corner_pts, camera_matrix, dist_coeffs, marker_size):
    """Pose of one marker via solvePnP. Returns (ok, rvec, tvec) in the camera frame."""
    half = marker_size / 2.0
    # 3D corners in the marker frame, same order as OpenCV returns them:
    # top-left, top-right, bottom-right, bottom-left
    object_points = np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float32,
    )
    image_points = corner_pts.reshape(4, 2).astype(np.float32)

    # IPPE returns the two possible solutions of the planar-pose ambiguity.
    # Polish both with Levenberg-Marquardt and keep the one that reprojects best.
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n == 0:
        return False, None, None

    best = None
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix,
                                      dist_coeffs, rv.copy(), tv.copy())
        projected, _ = cv2.projectPoints(object_points, rv, tv,
                                         camera_matrix, dist_coeffs)
        error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points,
                                             axis=1)))
        if best is None or error < best[0]:
            best = (error, rv, tv)
    return True, best[1], best[2]


def rotation_to_euler_deg(R):
    """Rotation matrix -> (roll, pitch, yaw) in degrees (rotations about x, y, z)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_text_block(img, lines, origin, color):
    """Draw several lines of text on a dark background so they stay readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad = 0.5, 1, 4
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + 6
    block_w = max(s[0] for s in sizes) + 2 * pad
    block_h = line_h * len(lines) + pad

    x, y = origin
    x = int(min(max(x, 0), max(img.shape[1] - block_w, 0)))
    y = int(min(max(y, 0), max(img.shape[0] - block_h, 0)))

    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), color, 1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x + pad, y + line_h * (i + 1) - 2),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main per-frame processing
# --------------------------------------------------------------------------- #
def process_frame(frame, detectors, camera_matrix, dist_coeffs, marker_size):
    """Detect markers, estimate and draw their poses. Returns a list of results."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []

    for family, detect in detectors.items():
        corners, ids, _ = detect(gray)
        if ids is None or len(ids) == 0:
            continue

        color = COLORS[family]
        aruco.drawDetectedMarkers(frame, corners, ids, color)

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = estimate_marker_pose(
                marker_corners, camera_matrix, dist_coeffs, marker_size)
            if not ok:
                continue

            # Align marker axes with the camera convention (see FLIP above)
            R, _ = cv2.Rodrigues(rvec)
            R_aligned = R @ FLIP
            rvec_aligned, _ = cv2.Rodrigues(R_aligned)
            roll, pitch, yaw = rotation_to_euler_deg(R_aligned)

            x, y, z = (tvec.flatten() * 100.0)          # metres -> cm
            distance = float(np.linalg.norm(tvec)) * 100.0

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec_aligned, tvec, marker_size * 0.6, 2)

            pts = marker_corners.reshape(4, 2)
            top_left = (pts[:, 0].min(), pts[:, 1].min() - 100)
            draw_text_block(frame, [
                f"{family} ID {int(marker_id)}",
                f"x {x:6.1f}  y {y:6.1f} cm",
                f"Depth z {z:6.1f} cm",
                f"Dist {distance:6.1f} cm",
                f"R {roll:6.1f} P {pitch:6.1f} Y {yaw:6.1f} deg",
            ], top_left, color)

            results.append({
                "family": family, "id": int(marker_id),
                "position_cm": (x, y, z), "depth_cm": z, "distance_cm": distance,
                "rpy_deg": (roll, pitch, yaw),
            })
    return results


def parse_args():
    p = argparse.ArgumentParser(description="ArUco pose estimation with a webcam")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--marker-size", type=float, default=0.05,
                   help="real side length of the black marker square in METRES "
                        "(default 0.05 = 5 cm)")
    p.add_argument("--calib", type=str, default=None,
                   help=".npz file with 'camera_matrix' and 'dist_coeffs'")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read a frame from the camera")

    camera_matrix, dist_coeffs, calibrated = load_camera_matrix(args.calib, frame.shape)
    if not calibrated:
        print("WARNING: no calibration file given - using an approximate camera "
              "model, so depth values are only rough estimates.")

    detectors = {name: build_detector(d) for name, d in DICTIONARIES.items()}

    prev_time = time.time()
    fps = 0.0
    shot_count = 0
    print("Running. Press 'q' or ESC to quit, 's' to save a screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = process_frame(frame, detectors, camera_matrix, dist_coeffs,
                                args.marker_size)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        status = "calibrated" if calibrated else "APPROX. camera model"
        cv2.putText(frame, f"FPS {fps:4.1f} | markers {len(results)} | {status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow("ArUco Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = f"aruco_pose_{shot_count:03d}.png"
            cv2.imwrite(name, frame)
            print("Saved", name)
            shot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
