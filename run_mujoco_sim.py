#!/usr/bin/env python3
"""
MuJoCo 3D Warehouse Corridor Simulation (v3 — Straight + Curved Turn + Obstacle Stop).
======================================================================================
What happens in this simulation:
  1. Robot starts in a straight warehouse rack aisle (plain matte tiled floor, zero markings).
  2. The aisle smoothly curves to the LEFT — the robot steers autonomously using ONLY
     OpenCV's steering angle output (`nav["steer_deg"]`) while maintaining its safety margin!
  3. Ahead after the turn, a fallen warehouse pallet/crate blocks the middle of the aisle —
     OpenCV detects `OBSTACLE AHEAD - STOP` and the robot automatically brakes before crashing!

Controls (click the OpenCV window to use):
    R        : Reset robot back to the start of the aisle
    Spacebar : Pause / Resume robot motion
    A / D    : Manual steering override (left / right)
    Q / Esc  : Quit

Tip for clean screen recording:
    Click the MuJoCo 3D window and press [Tab] and [Shift + Tab] to hide both side panels!
"""

import math
import time
import cv2
import mujoco
import mujoco.viewer

from detector import CorridorPathDetector


def aisle_centerline(x: float):
    """
    Defines the physical warehouse aisle centerline y(x) and tangent heading theta(x):
      - 0.0 m to 3.5 m : Straight aisle
      - 3.5 m to 9.5 m : Smooth left turn (+1.35 m lateral curve)
      - 9.5 m onward   : Straight exit leading to a mid-aisle obstacle crate
    """
    if x <= 3.5:
        return 0.0, 0.0
    elif x >= 9.5:
        return 1.35, 0.0
    else:
        u = (x - 3.5) / 6.0  # [0, 1]
        y = 1.35 * 0.5 * (1.0 - math.cos(math.pi * u))
        dydx = 1.35 * 0.5 * (math.pi / 6.0) * math.sin(math.pi * u)
        return y, math.atan(dydx)


def build_curved_warehouse_xml() -> str:
    """Builds a curved warehouse rack corridor with a matte tiled floor and a mid-aisle obstacle."""
    racks_xml = []
    aisle_half_width = 0.68
    rack_depth = 0.48

    xs = [1.0 + i * 1.15 for i in range(11)]
    for i, x in enumerate(xs):
        cy, theta = aisle_centerline(x)
        nx = -math.sin(theta)
        ny = math.cos(theta)

        for side_name, sign in (("L", 1.0), ("R", -1.0)):
            px_in = x + sign * aisle_half_width * nx
            py_in = cy + sign * aisle_half_width * ny

            px_out = x + sign * (aisle_half_width + rack_depth) * nx
            py_out = cy + sign * (aisle_half_width + rack_depth) * ny

            # Vertical blue rack uprights standing directly on the plain tiled floor
            racks_xml.append(
                f'<geom name="post_in_{side_name}{i}" type="cylinder" '
                f'pos="{px_in:.3f} {py_in:.3f} 0.60" size="0.065 0.60" rgba="0.10 0.36 0.70 1"/>'
            )
            racks_xml.append(
                f'<geom name="post_out_{side_name}{i}" type="cylinder" '
                f'pos="{px_out:.3f} {py_out:.3f} 0.60" size="0.055 0.60" rgba="0.10 0.30 0.58 1"/>'
            )

            # Horizontal orange shelf beams + storage boxes connecting consecutive bays
            if i < len(xs) - 1:
                x_next = xs[i + 1]
                cy_n, theta_n = aisle_centerline(x_next)
                nx_n, ny_n = -math.sin(theta_n), math.cos(theta_n)

                px_in_n = x_next + sign * aisle_half_width * nx_n
                py_in_n = cy_n + sign * aisle_half_width * ny_n

                bx = 0.5 * (px_in + px_in_n)
                by = 0.5 * (py_in + py_in_n)
                seg_len = math.hypot(px_in_n - px_in, py_in_n - py_in)
                seg_ang = math.atan2(py_in_n - py_in, px_in_n - px_in)

                mid_shelf_x = bx + sign * (rack_depth * 0.5) * (-math.sin(seg_ang))
                mid_shelf_y = by + sign * (rack_depth * 0.5) * (math.cos(seg_ang))

                for shelf_z in (0.50, 0.98):
                    racks_xml.append(
                        f'<geom name="beam_{side_name}{i}_{int(shelf_z*100)}" type="box" '
                        f'pos="{bx:.3f} {by:.3f} {shelf_z:.2f}" euler="0 0 {seg_ang:.4f}" '
                        f'size="{seg_len*0.48:.3f} 0.03 0.035" rgba="0.88 0.42 0.10 1"/>'
                    )
                    racks_xml.append(
                        f'<geom name="box_{side_name}{i}_{int(shelf_z*100)}" type="box" '
                        f'pos="{mid_shelf_x:.3f} {mid_shelf_y:.3f} {shelf_z + 0.14:.2f}" '
                        f'euler="0 0 {seg_ang:.4f}" '
                        f'size="0.24 0.17 0.13" rgba="0.65 0.50 0.32 1"/>'
                    )

    racks_block = "\n    ".join(racks_xml)

    # Mid-aisle obstacle crate placed after the curve (at x = 11.2 m, y = 1.35 m)
    obstacle_xml = """
    <geom name="mid_aisle_obstacle" type="box" pos="11.2 1.35 0.24" size="0.22 0.34 0.24" rgba="0.82 0.22 0.16 1"/>
    <geom name="obstacle_stripe" type="box" pos="10.97 1.35 0.24" size="0.01 0.28 0.08" rgba="0.95 0.85 0.15 1"/>
    """

    return f"""
<mujoco model="curved_warehouse_corridor">
  <visual>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.38 0.38 0.38" specular="0.05 0.05 0.05"/>
    <global offwidth="640" offheight="400"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.74 0.76 0.78" rgb2="0.60 0.62 0.65" width="512" height="512"/>
    <texture name="tile_tex" type="2d" builtin="checker" rgb1="0.84 0.84 0.82" rgb2="0.80 0.80 0.78"
             mark="edge" markrgb="0.68 0.68 0.66" width="512" height="512"/>
    <material name="tile_mat" texture="tile_tex" texrepeat="36 36" texuniform="true" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light pos="6 1 5" dir="0 0 -1" diffuse="0.72 0.72 0.72" castshadow="false"/>
    <geom name="tiled_floor" type="plane" pos="6 0.5 0" size="24 14 0.1" material="tile_mat"/>

    <!-- Distant warehouse back wall -->
    <geom name="back_wall" type="box" pos="15.0 1.0 1.5" size="0.1 8.0 1.5" rgba="0.76 0.77 0.78 1"/>

    <!-- Curved Warehouse Racks on Left and Right -->
    {racks_block}

    <!-- Mid-Aisle Obstacle Crate to demonstrate automatic safety stop -->
    {obstacle_xml}

    <!-- Mobile AMR Robot with front RGB camera -->
    <body name="amr_robot" pos="0 0 0.09">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <joint name="hinge_yaw" type="hinge" axis="0 0 1"/>

      <geom name="chassis" type="box" size="0.18 0.14 0.07" rgba="0.16 0.19 0.23 1"/>
      <geom name="wheel_l" type="cylinder" pos="0 0.155 -0.02" quat="0.707 0.707 0 0" size="0.07 0.02" rgba="0.08 0.08 0.08 1"/>
      <geom name="wheel_r" type="cylinder" pos="0 -0.155 -0.02" quat="0.707 0.707 0 0" size="0.07 0.02" rgba="0.08 0.08 0.08 1"/>
      <geom name="cam_housing" type="box" pos="0.16 0 0.16" size="0.03 0.05 0.03" rgba="0.05 0.05 0.05 1"/>

      <!-- Front RGB camera looking forward (+X), pitched 11 deg downward -->
      <camera name="robot_cam" pos="0.19 0 0.26" xyaxes="0 -1 0 0.191 0 0.982" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    xml_str = build_curved_warehouse_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=400, width=640)
    detector = CorridorPathDetector(ema_alpha=0.35)

    robot_x = -0.20
    robot_y = 0.0
    robot_yaw = 0.0
    paused = False

    print("Curved Warehouse Corridor + Closed-Loop OpenCV Steering & Obstacle Stop running!")
    print("Tip: Click MuJoCo window and press [Tab] and [Shift+Tab] to hide side panels.")
    print("Controls (click OpenCV window): [R] Reset | [Space] Pause | [A]/[D] Steer | [Q] Quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 0.0
        viewer.cam.elevation = -22.0
        viewer.cam.distance = 2.6
        viewer.cam.lookat[:] = [robot_x + 1.2, 0.0, 0.35]

        last_time = time.perf_counter()

        while viewer.is_running():
            now = time.perf_counter()
            dt = min(0.05, now - last_time)
            last_time = now

            # Update robot pose in MuJoCo and render front RGB camera
            data.qpos[0] = robot_x
            data.qpos[1] = robot_y
            data.qpos[2] = robot_yaw
            mujoco.mj_forward(model, data)

            viewer.cam.lookat[0] = robot_x + 1.1 * math.cos(robot_yaw)
            viewer.cam.lookat[1] = robot_y + 1.1 * math.sin(robot_yaw)
            viewer.cam.azimuth = math.degrees(robot_yaw)
            viewer.sync()

            renderer.update_scene(data, camera="robot_cam")
            rgb = renderer.render()
            raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Run decoupled classical OpenCV detector on raw BGR image
            vis, mask, nav = detector.process_frame(raw_bgr)

            # Closed-loop autonomous control driven ONLY by OpenCV's nav output!
            if not paused:
                if nav["stop"]:
                    # Automatic safety brake before crashing into obstacle / narrow pinch
                    linear_vel = 0.0
                    angular_vel = 0.0
                else:
                    linear_vel = 0.55
                    # Proportional steering controller: negative because +steer_deg means steer right (-yaw)
                    angular_vel = -0.022 * nav["steer_deg"]
                    angular_vel = max(-0.55, min(0.55, angular_vel))

                robot_yaw += angular_vel * dt
                robot_x += linear_vel * math.cos(robot_yaw) * dt
                robot_y += linear_vel * math.sin(robot_yaw) * dt

            cv2.imshow("Traversable Path Detection (OpenCV)", vis)
            cv2.imshow("Free-Space Floor Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                robot_x, robot_y, robot_yaw = -0.20, 0.0, 0.0
                detector = CorridorPathDetector(ema_alpha=0.35)
            elif key == ord("a"):
                robot_yaw += 0.06
            elif key == ord("d"):
                robot_yaw -= 0.06

    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()