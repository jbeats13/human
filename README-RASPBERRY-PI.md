# Human detector on Raspberry Pi

This project is set up to run on **Raspberry Pi** with:

- **Arducam / Raspberry Pi Camera Module 3** (CSI, libcamera)
- Optional **PCA9685** servo driver (e.g. [ArduCAM PCA9685](https://github.com/ArduCAM/PCA9685)) for pan-tilt or alerts

On a Raspberry Pi, the script automatically uses the Pi Camera (picamera2). Use a USB webcam instead with `--no-rpi-camera`.

---

## 1. Raspberry Pi OS

- Use **Raspberry Pi OS** (64-bit recommended for YOLO).
- Update: `sudo apt update && sudo apt upgrade -y`

---

## 2. Enable camera and I2C

```bash
sudo raspi-config
```

- **Interface Options → Camera** → Enable (for Arducam / Camera Module 3).
- **Interface Options → I2C** → Enable (for PCA9685 servo).

Reboot if prompted.

---

## 3. Dependencies

```bash
# System
sudo apt install -y python3-venv python3-pip
sudo apt install -y libopencv-dev python3-opencv # OpenCV
sudo apt install -y python3-smbus                # I2C for PCA9685

# picamera2 (usually pre-installed on Raspberry Pi OS)
sudo apt install -y python3-picamera2
```

Create a venv and install the project (from the repo root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install opencv-python-headless # if needed for display
```

---

## 4. Run the human detector

**Using the Pi Camera (default on Raspberry Pi):**

```bash
python human_detector.py
```

**Pan-tilt: scan until person, then track their center**

Use two servos (pan + tilt) on the PCA9685. When no person is in frame, the camera pans back and forth; when a person is detected, it tracks the center of the largest person. Typical Arducam pan-tilt: pan on channel 0, tilt on channel 1.

```bash
python human_detector.py --pan-channel 0 --tilt-channel 1
```

Tune scan/track with:

- `--pan-min 30 --pan-max 150` – pan range in degrees
- `--tilt-min 45 --tilt-max 135` – tilt range
- `--scan-speed 2` – pan degrees per frame while scanning
- `--track-gain 0.15` – higher = faster tracking (default 0.15)
- `--track-smoothing 0.25` – higher = smoother, less jitter (0–1)

**Single servo (two positions: human / no human):**

```bash
python human_detector.py --servo-channel 0
```

**Using a USB webcam instead of the Pi Camera:**

```bash
python human_detector.py --no-rpi-camera
```

**Headless (no display, e.g. over SSH):**

```bash
python human_detector.py --no-show
```

**Other options:**

- `--model yolo11n.pt` – small/fast (default); try `yolo26n.pt` for better accuracy.
- `--conf 0.5` – detection confidence threshold.
- `--rpi-camera-width 1280 --rpi-camera-height 720` – Pi camera resolution.
- `--servo-angle-detected 90 --servo-angle-none 0` – single-servo angles when human is / isn’t detected.

---

## 5. Hardware summary

| Component                 | Purpose                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| Raspberry Pi              | Host (Pi 4/5 recommended)                                            |
| Camera Module 3 / Arducam | CSI camera (picamera2)                                               |
| PCA9685                   | Optional; I2C servo driver (e.g. ArduCAM pan-tilt board)             |
| 2× servos                 | Pan + tilt for scan & track mode (channel 0 = pan, 1 = tilt typical) |

---

## 6. Troubleshooting

- **“Could not open camera”**
  - With Pi Camera: enable Camera in `raspi-config`, check cable and connector.
  - With USB: try `--no-rpi-camera` and ensure the webcam is not in use elsewhere.

- **“picamera2 not found”**
  - Install: `sudo apt install python3-picamera2`
  - Use USB camera: `python human_detector.py --no-rpi-camera`

- **“PCA9685 not available”**
  - Enable I2C in `raspi-config`.
  - Install: `sudo apt install python3-smbus` (or `pip install smbus2`).
  - Check wiring and I2C address (default 0x40).

- **Slow or low FPS**
  - Use `yolo11n.pt`, lower resolution (e.g. `--rpi-camera-width 640 --rpi-camera-height 480`), or a faster Pi (e.g. Pi 5).
