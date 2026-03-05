#!/usr/bin/env python3
"""
Human detector using webcam and YOLO.

Runs real-time detection on your camera feed and reports when a human (person) is visible.
Uses the COCO 'person' class (class id 0). Press 'q' to quit.

Camera options:
  - USB webcam: default (OpenCV).
  - Arducam / Raspberry Pi Camera Module 3: use --rpi-camera (uses picamera2 / libcamera).

Pan-tilt tracking (PCA9685):
  - With --pan-channel and --tilt-channel: servo scans back and forth until a person is
    detected, then tracks the center of the person (pan-tilt follows them).
  - With --servo-channel only: simple two-position mode (angle when human / no human).
https://github.com/ArduCAM/PCA9685
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root when ultralytics is the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from ultralytics import YOLO

# Optional: Raspberry Pi Camera Module 3 / Arducam (picamera2, libcamera)
try:
    from picamera2 import Picamera2

    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore
    PICAMERA2_AVAILABLE = False

# Optional PCA9685 for servo (Raspberry Pi + I2C)
try:
    from pca9685 import create_pca9685
except ImportError:
    create_pca9685 = None  # type: ignore

# COCO class 0 = person
PERSON_CLASS_ID = 0


def _box_center_xy(box) -> tuple[float, float]:
    """Return (center_x, center_y) of a YOLO box (xyxy format)."""
    xyxy = box.xyxy[0]
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return (x1 + x2) / 2, (y1 + y2) / 2


def _box_area(box) -> float:
    """Return area of a YOLO box (xyxy format)."""
    xyxy = box.xyxy[0]
    w = float(xyxy[2]) - float(xyxy[0])
    h = float(xyxy[3]) - float(xyxy[1])
    return w * h


def _is_raspberry_pi() -> bool:
    """True if running on a Raspberry Pi (for defaulting to RPi camera)."""
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode("utf-8", errors="ignore").lower()
            return "raspberry pi" in model
    except Exception:
        return False


def main(
    camera_id: int = 0,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.5,
    show_window: bool = True,
    use_rpi_camera: bool = False,
    rpi_camera_width: int = 1280,
    rpi_camera_height: int = 720,
    servo_channel: int | None = None,
    servo_angle_detected: float = 90.0,
    servo_angle_none: float = 0.0,
    pan_channel: int | None = None,
    tilt_channel: int | None = None,
    pan_min: float = 30.0,
    pan_max: float = 150.0,
    tilt_min: float = 45.0,
    tilt_max: float = 135.0,
    scan_speed: float = 2.0,
    track_gain: float = 0.15,
    track_smoothing: float = 0.25,
    pca9685_address: int = 0x40,
    pca9685_bus: int = 1,
):
    """Run human detection on the default camera.

    Pan-tilt tracking: when pan_channel and tilt_channel are set, the servos scan back and forth until a person is
    detected, then track the center of the person (largest detection).
    """
    track_mode = pan_channel is not None and tilt_channel is not None
    pca = None
    current_pan = (pan_min + pan_max) / 2 if track_mode else None
    current_tilt = (tilt_min + tilt_max) / 2 if track_mode else None
    scan_direction = 1.0  # +1 = pan increasing

    if track_mode or servo_channel is not None:
        if create_pca9685 is None:
            print("Warning: pca9685 module not found; servo disabled.")
        else:
            pca = create_pca9685(address=pca9685_address, bus_id=pca9685_bus)
            if pca is None:
                print("Warning: PCA9685 not available (I2C/smbus?). Servo disabled.")
                track_mode = False
            elif track_mode:
                print(
                    f"Pan-tilt tracking: pan ch {pan_channel}, tilt ch {tilt_channel}. Scan when no person, track center when person detected."
                )
            else:
                print(
                    f"Servo on PCA9685 channel {servo_channel}: angle {servo_angle_none}° (no human) / {servo_angle_detected}° (human)."
                )

    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    cap = None
    picam2 = None

    if use_rpi_camera:
        if not PICAMERA2_AVAILABLE:
            print("Error: --rpi-camera requires picamera2 (Raspberry Pi Camera / libcamera).")
            print("On Raspberry Pi OS: picamera2 is usually pre-installed.")
            return 1
        print("Opening Arducam / Raspberry Pi Camera Module 3 (picamera2)...")
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (rpi_camera_width, rpi_camera_height)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("preview")
        picam2.start()
        print(f"RPi camera started at {rpi_camera_width}x{rpi_camera_height}.")
    else:
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera. Check that it's connected and not in use.")
            return 1

    try:
        print("Human detector running. Press 'q' to quit.")
        print("Detecting only 'person' (human) in the frame.\n")

        while True:
            if picam2 is not None:
                frame = picam2.capture_array()
                if frame is None:
                    print("Failed to read frame from RPi camera.")
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame.")
                    break

            # Run detection, only for person class (class 0 in COCO)
            results = model.predict(
                frame,
                conf=confidence,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
            r = results[0]

            human_detected = r.boxes is not None and len(r.boxes) > 0
            count = len(r.boxes) if r.boxes else 0
            h, w = frame.shape[:2]
            frame_cx, frame_cy = w / 2, h / 2

            if pca is not None and track_mode:
                # Pan-tilt: scan when no person, track person center when detected
                if human_detected:
                    # Pick largest person and track their center
                    boxes_list = list(r.boxes)
                    largest = max(boxes_list, key=_box_area)
                    person_cx, person_cy = _box_center_xy(largest)
                    # Normalized error: -1 (left/up) to +1 (right/down)
                    err_x = (person_cx - frame_cx) / max(frame_cx, 1)
                    err_y = (person_cy - frame_cy) / max(frame_cy, 1)
                    # Target: move pan/tilt to bring person to center (positive err_x = person right of center → increase pan)
                    target_pan = current_pan + track_gain * err_x * (pan_max - pan_min) * 0.5
                    target_tilt = current_tilt + track_gain * err_y * (tilt_max - tilt_min) * 0.5
                    target_pan = max(pan_min, min(pan_max, target_pan))
                    target_tilt = max(tilt_min, min(tilt_max, target_tilt))
                    current_pan = current_pan + (1 - track_smoothing) * (target_pan - current_pan)
                    current_tilt = current_tilt + (1 - track_smoothing) * (target_tilt - current_tilt)
                    current_pan = max(pan_min, min(pan_max, current_pan))
                    current_tilt = max(tilt_min, min(tilt_max, current_tilt))
                    pca.setRotationAngle(pan_channel, current_pan)
                    pca.setRotationAngle(tilt_channel, current_tilt)
                    status = f"Tracking ({count} person(s))"
                else:
                    # Scan: sweep pan back and forth, keep tilt at center
                    current_pan += scan_speed * scan_direction
                    if current_pan >= pan_max:
                        current_pan = pan_max
                        scan_direction = -1.0
                    elif current_pan <= pan_min:
                        current_pan = pan_min
                        scan_direction = 1.0
                    pca.setRotationAngle(pan_channel, current_pan)
                    pca.setRotationAngle(tilt_channel, current_tilt)
                    status = "Scanning..."
            elif pca is not None:
                # Legacy single-servo mode
                if human_detected:
                    status = f"Human detected ({count} person(s))"
                    pca.setRotationAngle(servo_channel, servo_angle_detected)
                else:
                    status = "No human in frame"
                    pca.setRotationAngle(servo_channel, servo_angle_none)
            else:
                status = f"Human detected ({count} person(s))" if human_detected else "No human in frame"

            # Draw bounding boxes and labels on the frame
            annotated = r.plot()

            # In track mode, draw crosshair at frame center (where we aim)
            if track_mode and pca is not None:
                cx, cy = int(frame_cx), int(frame_cy)
                size = 20
                cv2.line(annotated, (cx - size, cy), (cx + size, cy), (0, 255, 255), 2)
                cv2.line(annotated, (cx, cy - size), (cx, cy + size), (0, 255, 255), 2)

            # Show status on the image
            cv2.putText(
                annotated,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if human_detected else (0, 0, 255),
                2,
            )

            if show_window:
                cv2.imshow("Human detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless: just print status every frame (can be noisy)
                print(f"\r{status}", end="")

        return 0
    finally:
        if cap is not None:
            cap.release()
        if picam2 is not None:
            picam2.stop()
        if show_window:
            cv2.destroyAllWindows()
        if pca is not None:
            pca.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect humans in the camera feed.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model name (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window (print status only)",
    )
    _default_rpi_camera = _is_raspberry_pi() and PICAMERA2_AVAILABLE
    parser.add_argument(
        "--rpi-camera",
        action="store_true",
        default=_default_rpi_camera,
        help="Use Arducam / Raspberry Pi Camera Module 3 (default: on when on Raspberry Pi)",
    )
    parser.add_argument(
        "--no-rpi-camera",
        action="store_true",
        help="Use USB webcam instead of RPi camera (when on Raspberry Pi)",
    )
    parser.add_argument(
        "--rpi-camera-width",
        type=int,
        default=1280,
        help="RPi camera frame width (default: 1280)",
    )
    parser.add_argument(
        "--rpi-camera-height",
        type=int,
        default=720,
        help="RPi camera frame height (default: 720)",
    )
    parser.add_argument(
        "--servo-channel",
        type=int,
        default=None,
        metavar="N",
        help="PCA9685 channel (0-15) for single servo; move when human detected/not",
    )
    parser.add_argument(
        "--servo-angle-detected",
        type=float,
        default=90.0,
        help="Servo angle (0-180) when human detected (default: 90)",
    )
    parser.add_argument(
        "--servo-angle-none",
        type=float,
        default=0.0,
        help="Servo angle (0-180) when no human (default: 0)",
    )
    parser.add_argument(
        "--pan-channel",
        type=int,
        default=None,
        metavar="N",
        help="PCA9685 channel for pan servo (with --tilt-channel: scan then track person center)",
    )
    parser.add_argument(
        "--tilt-channel",
        type=int,
        default=None,
        metavar="N",
        help="PCA9685 channel for tilt servo (with --pan-channel: scan then track)",
    )
    parser.add_argument(
        "--pan-min",
        type=float,
        default=30.0,
        help="Pan servo min angle in degrees (default: 30)",
    )
    parser.add_argument(
        "--pan-max",
        type=float,
        default=150.0,
        help="Pan servo max angle in degrees (default: 150)",
    )
    parser.add_argument(
        "--tilt-min",
        type=float,
        default=45.0,
        help="Tilt servo min angle in degrees (default: 45)",
    )
    parser.add_argument(
        "--tilt-max",
        type=float,
        default=135.0,
        help="Tilt servo max angle in degrees (default: 135)",
    )
    parser.add_argument(
        "--scan-speed",
        type=float,
        default=2.0,
        help="Pan scan speed in degrees per frame when no person (default: 2)",
    )
    parser.add_argument(
        "--track-gain",
        type=float,
        default=0.15,
        help="Tracking responsiveness 0-1 (default: 0.15)",
    )
    parser.add_argument(
        "--track-smoothing",
        type=float,
        default=0.25,
        help="Tracking smoothing 0-1, higher = smoother (default: 0.25)",
    )
    parser.add_argument(
        "--pca9685-address",
        type=lambda x: int(x, 0),
        default=0x40,
        help="PCA9685 I2C address (default: 0x40)",
    )
    parser.add_argument(
        "--pca9685-bus",
        type=int,
        default=1,
        help="I2C bus number (default: 1)",
    )
    args = parser.parse_args()

    use_rpi = args.rpi_camera and not args.no_rpi_camera
    sys.exit(
        main(
            camera_id=args.camera,
            model_name=args.model,
            confidence=args.conf,
            show_window=not args.no_show,
            use_rpi_camera=use_rpi,
            rpi_camera_width=args.rpi_camera_width,
            rpi_camera_height=args.rpi_camera_height,
            servo_channel=args.servo_channel,
            servo_angle_detected=args.servo_angle_detected,
            servo_angle_none=args.servo_angle_none,
            pan_channel=args.pan_channel,
            tilt_channel=args.tilt_channel,
            pan_min=args.pan_min,
            pan_max=args.pan_max,
            tilt_min=args.tilt_min,
            tilt_max=args.tilt_max,
            scan_speed=args.scan_speed,
            track_gain=args.track_gain,
            track_smoothing=args.track_smoothing,
            pca9685_address=args.pca9685_address,
            pca9685_bus=args.pca9685_bus,
        )
    )
