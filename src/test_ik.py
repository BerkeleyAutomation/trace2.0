"""
Interactive IK testing script.

Prompts for [x, y, z] world-coordinate positions (meters), then moves the
robot to that position using either linear or cartesian motion.

Example positions (from perform_push_through):
  Right-arm workspace:  [0.45, -0.15, 0.19]    (hover height)
  Left-arm workspace:   [0.45,  0.15, 0.19]    (hover height)
  Table-level depth:    z ≈ 0.18  (FOAM_DEPTH_R / FOAM_DEPTH_L)
  Safe height:          z = 0.30  (SAFE_HOME_Z)

Usage:
    python test_ik.py
"""

import sys
import numpy as np

sys.path.append('/home/justinyu/multicable-decluttering/')
sys.path.append('/home/justinyu/multicable-decluttering/yumi_jacobi')

from autolab_core import RigidTransform
from yumi_jacobi.interface import Interface

# ── Constants (from motion_jacobi.py / run_constants.py) ──────────────────
IFACE_SPEED = 0.26
SAFE_HOME_Z = 0.3

YUMI_MIN_POS = [-2.94, -2.350, -2.94, -2.16, -5.00, -1.54, -3.99]
YUMI_MAX_POS = [2.9409, 0.7592, 2.9409, 1.2, 5.00, 2.4086, 3.8]

# TCP frame names (must match yumi_jacobi interface)
R_TCP_FRAME = "r_tcp_frame"
L_TCP_FRAME = "l_tcp_frame"

# Gripper-down rotation matrices
GRIP_DOWN_R = np.diag([1, -1, -1]).astype(float)
GRIP_DOWN_L = np.diag([-1, 1, -1]).astype(float)

# ── Preset positions (representative of perform_push_through) ─────────────
PRESETS = {
    "1": ("Right hover",       [0.45, -0.15, 0.25], "right"),
    "2": ("Right table",       [0.45, -0.15, 0.184], "right"),
    "3": ("Left hover",        [0.45,  0.15, 0.25], "left"),
    "4": ("Left table",        [0.45,  0.15, 0.183], "left"),
    "5": ("Center safe",       [0.40,  0.00, 0.30], "left"),
    "6": ("Right far forward", [0.50, -0.20, 0.22], "right"),
    "7": ("Left far forward",  [0.50,  0.20, 0.22], "left"),
}


def pick_arm(y_coord: float) -> str:
    """Choose arm based on y-coordinate (negative-y → right arm)."""
    return "right" if y_coord < 0 else "left"


def build_transform(translation, arm: str) -> RigidTransform:
    """Build a RigidTransform for the given arm pointing straight down."""
    if arm == "right":
        return RigidTransform(
            translation=np.array(translation, dtype=float),
            rotation=GRIP_DOWN_R,
            from_frame=R_TCP_FRAME,
            to_frame="base_link",
        )
    else:
        return RigidTransform(
            translation=np.array(translation, dtype=float),
            rotation=GRIP_DOWN_L,
            from_frame=L_TCP_FRAME,
            to_frame="base_link",
        )


def move_robot(iface: Interface, target_tf: RigidTransform, arm: str, mode: str):
    """Execute linear or cartesian motion for one arm."""
    print(f"\n  Moving {arm} arm → {target_tf.translation}  (mode={mode})")
    if mode == "l":
        # Linear motion
        if arm == "right":
            iface.go_linear_single(r_target=target_tf)
        else:
            iface.go_linear_single(l_target=target_tf)
    else:
        # Cartesian (free-space) motion
        if arm == "right":
            iface.go_cartesian_waypoints(r_targets=[target_tf])
        else:
            iface.go_cartesian_waypoints(l_targets=[target_tf])
    print("  Done.")


def print_fk(iface: Interface):
    """Print current FK for both arms."""
    for arm in ("left", "right"):
        fk = iface.get_FK(arm)
        print(f"  {arm:>5} FK: translation={np.round(fk.translation, 4)}")


def print_menu():
    print("\n" + "=" * 60)
    print("  TEST IK — Interactive Robot Position Tester")
    print("=" * 60)
    print("  Presets:")
    for key, (name, pos, arm) in PRESETS.items():
        print(f"    [{key}] {name:25s} {pos}  ({arm})")
    print()
    print("  Or enter a custom position:  x y z       (e.g. 0.45 -0.15 0.20)")
    print("  Or enter a custom position:  x y z arm   (e.g. 0.45 -0.15 0.20 right)")
    print()
    print("  Commands:")
    print("    fk    — print current FK for both arms")
    print("    home  — move to home position")
    print("    q     — quit")
    print("-" * 60)


def parse_position(user_input: str):
    """
    Parse user input into (translation, arm).
    Returns (None, None) on failure.
    """
    # Check presets
    if user_input.strip() in PRESETS:
        _, pos, arm = PRESETS[user_input.strip()]
        return pos, arm

    # Try custom format:  x y z  [arm]
    parts = user_input.strip().split()
    if len(parts) < 3:
        return None, None

    try:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None, None

    if len(parts) >= 4 and parts[3] in ("left", "right"):
        arm = parts[3]
    else:
        arm = pick_arm(y)

    return [x, y, z], arm


def main():
    print("Initializing robot interface...")
    iface = Interface(IFACE_SPEED)
    iface.yumi.left.min_position = YUMI_MIN_POS
    iface.yumi.right.min_position = YUMI_MIN_POS
    iface.yumi.left.max_position = YUMI_MAX_POS
    iface.yumi.right.max_position = YUMI_MAX_POS

    print("Homing...")
    iface.home()
    iface.calibrate_grippers()
    iface.close_grippers()

    print("Ready!\n")
    print_fk(iface)

    while True:
        print_menu()
        user_input = input(">>> ").strip()

        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────────
        if user_input.lower() == "q":
            print("Homing and exiting...")
            iface.home()
            break

        if user_input.lower() == "fk":
            print_fk(iface)
            continue

        if user_input.lower() == "home":
            print("Homing...")
            iface.home()
            print_fk(iface)
            continue

        # ── Parse position ────────────────────────────────────────
        pos, arm = parse_position(user_input)
        if pos is None:
            print("  *** Invalid input. Enter a preset number, 'x y z', or 'x y z arm'.")
            continue

        print(f"\n  Target: {pos}  (arm={arm})")

        # ── Ask for motion type ───────────────────────────────────
        mode_input = input("  Motion type — [l]inear / [c]artesian (default=c): ").strip().lower()
        if mode_input not in ("l", "c", ""):
            print("  *** Invalid choice, defaulting to cartesian.")
            mode_input = "c"
        if mode_input == "":
            mode_input = "c"

        # ── Execute ───────────────────────────────────────────────
        target_tf = build_transform(pos, arm)
        try:
            move_robot(iface, target_tf, arm, mode_input)
        except Exception as e:
            print(f"  *** Motion FAILED: {e}")
            print("  Attempting recovery (homing)...")
            try:
                iface.home()
            except Exception:
                print("  *** Home also failed. Robot may need manual recovery.")

        print_fk(iface)


if __name__ == "__main__":
    main()
