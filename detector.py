#!/usr/bin/env python3
"""
Standalone Classical OpenCV Traversable Path Detector (v4.2 — Hybrid Rack Envelope + Obstacle Stop).
====================================================================================================
Features:
  - Self-supervised LAB/HSV plain floor modeling (zero floor markings required).
  - Inner rack-pillar base detection + perspective-bounded Left & Right rack boundary walls
    (prevents driving underneath open lower shelves).
  - Row-wise floor-connected free-space scanning with 14% safety clearance margin and
    2nd-order curved centerline fitting for smooth turns.
  - Accurate distinction between mid-aisle OBSTACLE AHEAD - STOP and PASSAGE TOO NARROW - STOP.
"""

import argparse
import math
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np


class CorridorPathDetector:
    def __init__(
        self,
        ema_alpha: float = 0.35,
        safety_margin_ratio: float = 0.14,   # 14% clearance buffer on each side of the aisle
        min_safe_width_bottom_px: int = 230, # Minimum clear width (px) at bottom for robot to fit
        min_lookahead_rows: int = 155,       # Safe braking distance in image rows ahead of robot
    ):
        self.alpha = ema_alpha
        self.safety_margin_ratio = safety_margin_ratio
        self.min_safe_width_bottom_px = min_safe_width_bottom_px
        self.min_lookahead_rows = min_lookahead_rows

        self.prev_steer_deg = 0.0
        self.prev_offset_px = 0.0
        self.prev_left_line = None   # (m, b)
        self.prev_right_line = None  # (m, b)

    @staticmethod
    def _filter_inner_bases(
        pts: List[Tuple[int, int]], side: str, y_bin: int = 28
    ) -> List[Tuple[int, int]]:
        """
        When front and rear rack posts appear at similar depths (y), keep ONLY the
        inner aisle-facing post in each depth band (largest x on LEFT, smallest x on RIGHT).
        """
        if not pts:
            return []
        bins: Dict[int, Tuple[int, int]] = {}
        for (px, py) in pts:
            b_idx = py // y_bin
            if b_idx not in bins:
                bins[b_idx] = (px, py)
            else:
                cur_x, _ = bins[b_idx]
                if (side == "LEFT" and px > cur_x) or (side == "RIGHT" and px < cur_x):
                    bins[b_idx] = (px, py)
        return sorted(bins.values(), key=lambda p: p[1])

    @staticmethod
    def _fit_rack_envelope(
        base_pts: List[Tuple[int, int]],
        side: str,
        prev_line,
        default_m: float,
        default_b: float,
        ema_alpha: float = 0.30,
    ) -> Tuple[float, float]:
        """
        Fits a perspective-valid boundary x = m*y + b connecting inner rack pillar bases.
        """
        m_est, b_est = None, None

        if len(base_pts) >= 2:
            ys = [p[1] for p in base_pts]
            if max(ys) - min(ys) >= 35:
                pts = np.array(base_pts, dtype=np.float32)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
                if abs(vy) > 1e-3:
                    cand_m = float(vx / vy)
                    cand_b = float(x0 - cand_m * y0)
                    valid_left = (side == "LEFT" and -2.2 <= cand_m <= -0.15)
                    valid_right = (side == "RIGHT" and 0.15 <= cand_m <= 2.2)
                    if valid_left or valid_right:
                        m_est, b_est = cand_m, cand_b

        if m_est is None and len(base_pts) >= 1:
            px, py = base_pts[-1]
            m_est = prev_line[0] if prev_line is not None else default_m
            b_est = px - m_est * py

        if m_est is not None and b_est is not None:
            if prev_line is not None:
                m_est = ema_alpha * m_est + (1.0 - ema_alpha) * prev_line[0]
                b_est = ema_alpha * b_est + (1.0 - ema_alpha) * prev_line[1]
            return m_est, b_est

        if prev_line is not None:
            return prev_line
        return default_m, default_b

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Processes a raw BGR image frame using classical OpenCV only.
        Returns: (annotated_frame, debug_mask, nav_dict)
        """
        t0 = time.perf_counter()
        frame = cv2.resize(frame, (800, 500))
        h, w = frame.shape[:2]
        vis = frame.copy()

        # ------------------------------------------------------------------
        # STEP 1: Bilateral Filter (suppresses tile grout, keeps pillar edges)
        # ------------------------------------------------------------------
        smooth = cv2.bilateralFilter(frame, d=7, sigmaColor=35, sigmaSpace=35)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ------------------------------------------------------------------
        # STEP 2: Self-Supervised Plain Floor Model (bottom-center seed patch)
        # ------------------------------------------------------------------
        ref_lab = lab[int(h * 0.85):int(h * 0.97), int(w * 0.42):int(w * 0.58)]
        ref_hsv = hsv[int(h * 0.85):int(h * 0.97), int(w * 0.42):int(w * 0.58)]

        med_l = float(np.median(ref_lab[:, :, 0]))
        med_a = float(np.median(ref_lab[:, :, 1]))
        med_b = float(np.median(ref_lab[:, :, 2]))
        med_s = float(np.median(ref_hsv[:, :, 1]))

        da = np.abs(lab[:, :, 1] - med_a)
        db = np.abs(lab[:, :, 2] - med_b)
        ds = np.maximum(0.0, hsv[:, :, 1] - med_s)
        dl = np.abs(lab[:, :, 0] - med_l)

        floor_dist = 1.60 * np.hypot(da, db) + 1.25 * ds + 0.35 * dl
        floor_raw = (floor_dist < 38.0).astype(np.uint8) * 255

        y_ceil_cut = int(h * 0.38)
        floor_raw[:y_ceil_cut, :] = 0

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        floor_clean = cv2.morphologyEx(floor_raw, cv2.MORPH_CLOSE, k_close)
        floor_clean = cv2.morphologyEx(floor_clean, cv2.MORPH_OPEN, k_open)

        # ------------------------------------------------------------------
        # STEP 3: Layer 1 — Detect Inner Rack Pillars & Build Rack Walls
        # ------------------------------------------------------------------
        non_floor = cv2.bitwise_not(floor_clean)
        non_floor[:y_ceil_cut, :] = 0
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
        pillar_mask = cv2.morphologyEx(non_floor, cv2.MORPH_OPEN, k_vert)

        contours, _ = cv2.findContours(pillar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_left_bases = []
        raw_right_bases = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 120:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            y_base = y + ch
            x_center = x + cw // 2

            if ch < cw * 1.4:
                continue
            if y_base >= h - 5 or x <= 2 or x + cw >= w - 3:
                continue

            if x_center < w * 0.48:
                raw_left_bases.append((x + cw, y_base))
            elif x_center > w * 0.52:
                raw_right_bases.append((x, y_base))

        left_bases = self._filter_inner_bases(raw_left_bases, "LEFT")
        right_bases = self._filter_inner_bases(raw_right_bases, "RIGHT")

        for (bx, by) in left_bases + right_bases:
            cv2.circle(vis, (bx, by), 5, (255, 220, 0), -1, cv2.LINE_AA)

        m_l, b_l = self._fit_rack_envelope(left_bases, "LEFT", self.prev_left_line, default_m=-0.95, default_b=520.0)
        m_r, b_r = self._fit_rack_envelope(right_bases, "RIGHT", self.prev_right_line, default_m=0.95, default_b=280.0)
        self.prev_left_line = (m_l, b_l)
        self.prev_right_line = (m_r, b_r)

        # ------------------------------------------------------------------
        # STEP 4: Layer 2 — Clip Floor to Rack Corridor & Find Connected Space
        # ------------------------------------------------------------------
        corridor_floor = floor_clean.copy()
        for y in range(y_ceil_cut, h):
            xl_wall = int(np.clip(m_l * y + b_l, 0, int(w * 0.45)))
            xr_wall = int(np.clip(m_r * y + b_r, int(w * 0.55), w - 1))
            if xl_wall > 0:
                corridor_floor[y, :xl_wall] = 0
            if xr_wall < w - 1:
                corridor_floor[y, xr_wall + 1:] = 0

        num_labels, labels = cv2.connectedComponents(corridor_floor)
        seed_y, seed_x = int(h * 0.93), w // 2
        seed_label = labels[seed_y, seed_x]
        if seed_label == 0:
            for dx in range(-40, 41, 10):
                if labels[seed_y, w // 2 + dx] > 0:
                    seed_label = labels[seed_y, w // 2 + dx]
                    break

        if seed_label > 0:
            free_floor = (labels == seed_label).astype(np.uint8) * 255
        else:
            free_floor = np.zeros((h, w), dtype=np.uint8)

        # ------------------------------------------------------------------
        # STEP 5: Row-Wise Scanning, Safety Margin & Obstacle / Width Check
        # ------------------------------------------------------------------
        y_start = h - 4
        y_limit = int(h * 0.42)
        step_y = 6

        raw_left_pts = []
        raw_right_pts = []
        safe_left_pts = []
        safe_right_pts = []
        center_pts = []

        narrow_detected = False
        obstacle_ahead = False
        stop_row_y = y_limit

        curr_cx = w // 2

        for y in range(y_start, y_limit, -step_y):
            row = free_floor[y, :]

            if row[curr_cx] == 0:
                found_x = None
                for offset in range(6, 20, 6):
                    if curr_cx + offset < w and row[curr_cx + offset] > 0:
                        found_x = curr_cx + offset
                        break
                    if curr_cx - offset >= 0 and row[curr_cx - offset] > 0:
                        found_x = curr_cx - offset
                        break
                if found_x is None:
                    stop_row_y = y
                    break
                curr_cx = found_x

            lx = curr_cx
            while lx > 0 and row[lx - 1] > 0:
                lx -= 1

            rx = curr_cx
            while rx < w - 1 and row[rx + 1] > 0:
                rx += 1

            row_width = rx - lx
            persp_scale = max(0.24, (y - y_ceil_cut) / float(h - y_ceil_cut))
            min_req_width = self.min_safe_width_bottom_px * persp_scale

            if row_width < min_req_width:
                narrow_detected = True
                stop_row_y = y
                break

            margin = int(max(8, row_width * self.safety_margin_ratio))
            safe_lx = lx + margin
            safe_rx = rx - margin

            if safe_rx <= safe_lx + 12:
                narrow_detected = True
                stop_row_y = y
                break

            curr_cx = (safe_lx + safe_rx) // 2

            raw_left_pts.append([lx, y])
            raw_right_pts.append([rx, y])
            safe_left_pts.append([safe_lx, y])
            safe_right_pts.append([safe_rx, y])
            center_pts.append([curr_cx, y])

        # Distinguish solid mid-aisle OBSTACLE from symmetrical NARROW PASSAGE:
        # Check the corridor center 12 px above stop_row_y — if blocked, it's an obstacle!
        free_depth_rows = y_start - stop_row_y
        if free_depth_rows < self.min_lookahead_rows:
            check_y = max(y_ceil_cut + 2, stop_row_y - 12)
            mid_band = free_floor[check_y, max(0, curr_cx - 45):min(w, curr_cx + 45)]
            center_blocked_ratio = 1.0 - (float(np.count_nonzero(mid_band)) / max(1, mid_band.size))
            if center_blocked_ratio > 0.45 or not narrow_detected:
                obstacle_ahead = True
                narrow_detected = False

        # ------------------------------------------------------------------
        # STEP 6: Curved Centerline Fitting & Steering / Brake Decision
        # ------------------------------------------------------------------
        steer_deg = 0.0
        offset_px = 0.0
        curve_pts = []

        if len(center_pts) >= 5:
            c_arr = np.array(center_pts, dtype=np.float32)
            cy = c_arr[:, 1]
            cx = c_arr[:, 0]

            coeffs = np.polyfit(cy, cx, deg=2)
            y_eval = np.linspace(cy.min(), cy.max(), 25)
            x_eval = np.polyval(coeffs, y_eval)
            curve_pts = np.stack([np.clip(x_eval, 0, w - 1), y_eval], axis=1).astype(np.int32)

            look_idx = min(len(curve_pts) - 1, max(0, int(len(curve_pts) * 0.30)))
            target_x, target_y = curve_pts[look_idx]
            bot_x = float(np.polyval(coeffs, float(y_start)))

            offset_px = bot_x - (w * 0.5)
            steer_deg = math.degrees(math.atan2(float(target_x) - (w * 0.5), max(1.0, float(y_start - target_y))))

        self.prev_steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        self.prev_offset_px = self.alpha * offset_px + (1.0 - self.alpha) * self.prev_offset_px

        should_stop = False
        if len(safe_left_pts) < 4:
            status_text = "BLOCKED - EMERGENCY STOP"
            status_color = (0, 0, 255)
            should_stop = True
        elif obstacle_ahead:
            status_text = "OBSTACLE AHEAD - STOP"
            status_color = (0, 60, 255)
            should_stop = True
        elif narrow_detected and free_depth_rows < self.min_lookahead_rows:
            status_text = "PASSAGE TOO NARROW - STOP"
            status_color = (0, 120, 255)
            should_stop = True
        elif abs(self.prev_steer_deg) > 6.0:
            direction = "RIGHT" if self.prev_steer_deg > 0 else "LEFT"
            status_text = f"CURVE {direction} - TRACKING"
            status_color = (0, 255, 200)
        else:
            status_text = "CLEAR PATH - SAFE"
            status_color = (0, 255, 0)

        # ------------------------------------------------------------------
        # STEP 7: Visualization
        # ------------------------------------------------------------------
        overlay = vis.copy()

        if len(raw_left_pts) >= 2:
            outer_poly = np.array(raw_left_pts + raw_right_pts[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [outer_poly], (0, 140, 220))

            safe_poly = np.array(safe_left_pts + safe_right_pts[::-1], dtype=np.int32)
            fill_col = (0, 195, 0) if not should_stop else (0, 80, 235)
            cv2.fillPoly(overlay, [safe_poly], fill_col)
            cv2.addWeighted(overlay, 0.36, vis, 0.64, 0, vis)

            # Outer rack boundaries (Red) & inner safety margins (Cyan)
            cv2.polylines(vis, [np.array(raw_left_pts, dtype=np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.polylines(vis, [np.array(raw_right_pts, dtype=np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.polylines(vis, [np.array(safe_left_pts, dtype=np.int32)], False, (255, 230, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis, [np.array(safe_right_pts, dtype=np.int32)], False, (255, 230, 0), 2, cv2.LINE_AA)

            if obstacle_ahead or narrow_detected:
                top_l = tuple(raw_left_pts[-1])
                top_r = tuple(raw_right_pts[-1])
                cv2.line(vis, top_l, top_r, (0, 0, 255), 4, cv2.LINE_AA)
                label = "OBSTACLE STOP LINE" if obstacle_ahead else "TOO NARROW"
                cv2.putText(
                    vis,
                    label,
                    (max(10, (top_l[0] + top_r[0]) // 2 - 95), max(92, top_l[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 60, 255),
                    2,
                    cv2.LINE_AA,
                )

        if len(curve_pts) >= 2 and not should_stop:
            cv2.polylines(vis, [curve_pts], False, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.arrowedLine(
                vis,
                tuple(curve_pts[min(3, len(curve_pts) - 1)]),
                tuple(curve_pts[0]),
                (0, 255, 255),
                3,
                cv2.LINE_AA,
                tipLength=0.35,
            )

        fps = 1.0 / max(1e-5, time.perf_counter() - t0)
        cv2.rectangle(vis, (8, 8), (540, 82), (20, 24, 28), -1)
        cv2.putText(vis, f"STATUS: {status_text}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"Steer: {self.prev_steer_deg:+.1f} deg | Offset: {self.prev_offset_px:+.0f} px | FPS: {fps:.1f}",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 240, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "Green: Safe Path | Amber: Safety Margin | Red: Rack Wall / Stop",
            (18, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (175, 210, 225),
            1,
            cv2.LINE_AA,
        )

        nav = {
            "status": status_text,
            "steer_deg": float(self.prev_steer_deg),
            "lateral_offset_px": float(self.prev_offset_px),
            "stop": should_stop,
        }
        return vis, free_floor, nav


def main():
    parser = argparse.ArgumentParser(description="Standalone OpenCV Traversable Path Detector (v4.2)")
    parser.add_argument("--input", type=str, default=None, help="Path to an MP4/AVI video file")
    parser.add_argument("--webcam", action="store_true", help="Run live using webcam")
    args = parser.parse_args()

    source = 0 if args.webcam else args.input
    if source is None:
        print("Usage: python detector.py --input video.mp4  OR  python detector.py --webcam")
        return

    cap = cv2.VideoCapture(source)
    detector = CorridorPathDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vis, mask, _ = detector.process_frame(frame)
        cv2.imshow("Traversable Path Detection (OpenCV)", vis)
        cv2.imshow("Free-Space Floor Mask", mask)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()