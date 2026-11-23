# Real-Time Traversable Path Detection Between Warehouse Racks (OpenCV)

A classical OpenCV computer vision system that detects and visualizes the traversable corridor between warehouse racks on a plain tiled floor—without relying on floor markings, fiducial tags, or predefined floor features.

To test and validate the perception pipeline end-to-end without physical warehouse hardware, the repository includes a **decoupled 3D MuJoCo warehouse corridor simulation** featuring a straight-to-curved aisle, open lower rack bays, and a mid-aisle obstacle crate. The detector receives **only the raw RGB camera frame** and has zero access to simulator coordinates.

## Key Capabilities

* **No Floor Markings Required**: Dynamically learns the plain tiled floor appearance from a bottom-center reference patch directly in front of the robot using `LAB` and `HSV` color spaces, while suppressing tile grout lines via `cv2.bilateralFilter`.
* **Two-Layer Hybrid Corridor Architecture**:
  1. **Rack Boundary Envelope**: Isolates tall vertical rack uprights (filtering out rear rack posts and wide crates) and fits perspective-bounded Left & Right rack walls (`cv2.fitLine` with `cv2.DIST_HUBER`) so the robot never tries to drive underneath open lower shelves.
  2. **Connected-Component Free-Space Scanner**: Clips the floor mask inside the rack walls, extracts robot-reachable floor via `cv2.connectedComponents`, and scans upward row-by-row.
* **Curved Path Tracking**: Fits a 2nd-order polynomial (`np.polyfit`) to the row-wise corridor midpoints to track both straight aisles and curved turns, outputting a real-time steering angle (`steer_deg`) and lateral offset (`px`).
* **Safety Clearance Margin & Narrow-Passage Check**: Applies a 14% inward clearance buffer on both sides of the corridor (amber buffer zone vs. green safe path) and checks perspective-scaled minimum passable width (`PASSAGE TOO NARROW - STOP`).
* **Automatic Obstacle Braking**: Detects when a mid-aisle obstacle blocks the free-space lookahead and triggers `OBSTACLE AHEAD - STOP` with a red stop barrier line, braking the simulated AMR before collision.

## Repository Structure

```text
.
├── detector.py          # Standalone classical OpenCV traversable path detector (works on sim, video, or webcam)
├── run_mujoco_sim.py    # Decoupled MuJoCo 3D curved warehouse rack corridor + closed-loop AMR controller
├── requirements.txt     # Python dependencies
├── demo.mp4             # Recorded split-screen simulation & OpenCV detection demo (optional)
└── README.md
```

## Setup & Installation

Tested on **Python 3.12 (64-bit)** on Windows 11.

```bash
pip install -r requirements.txt
```

*(Windows note: If `mujoco.dll` fails to load on a fresh Windows install, install the official [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vc14/vc_redist.x64.exe).)*

## How to Run

### 1. Run the 3D MuJoCo Warehouse Simulation + Live OpenCV Detector
```bash
python run_mujoco_sim.py
```

**Controls (click the OpenCV window to focus):**
* `R` — Reset robot back to the start of the aisle
* `Spacebar` — Pause / resume robot motion
* `A` / `D` — Manual left / right steering override
* `Q` or `Esc` — Quit
* *(In the MuJoCo 3D window, press `Tab` and `Shift + Tab` to hide the sidebars.)*

### 2. Run the Standalone Detector on Any Real Video or Webcam
```bash
# On a recorded warehouse or PVC-pipe corridor video
python detector.py --input path/to/video.mp4

# Live webcam feed
python detector.py --webcam
```