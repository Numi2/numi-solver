#!/usr/bin/env python3
"""Report actual release, descent, and support from a native Metal fruit trace."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def audit(path: Path, expected_frames: int, fruit_count: int) -> dict:
    payload = path.read_bytes()
    with io.StringIO(payload.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "frame", "time_s", "fruit", "x_m", "y_m", "z_m", "radius_m",
            "vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s",
            "last_substep_ground_impulse_Ns", "released",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("trace columns do not match the native fruit export")
        frames = {}
        for row in reader:
            frame, fruit = int(row["frame"]), int(row["fruit"])
            values = {name: float(row[name]) for name in required - {"frame", "fruit"}}
            if frame < 0 or not 0 <= fruit < fruit_count:
                raise ValueError("invalid frame or fruit index")
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite native fruit state")
            if values["radius_m"] <= 0 or values["released"] not in (0, 1):
                raise ValueError("invalid radius or release bit")
            if values["last_substep_ground_impulse_Ns"] < 0:
                raise ValueError("negative unilateral support impulse")
            if fruit in frames.setdefault(frame, {}):
                raise ValueError("duplicate fruit in a native frame")
            frames[frame][fruit] = values
    if not frames or sorted(frames) != list(range(max(frames) + 1)):
        raise ValueError("trace must contain consecutive frames starting at zero")
    # A file copied while its producer is writing may end inside the last frame.
    # Retain only the verified complete prefix; never treat that as termination.
    last = max(frames)
    partial_tail = len(frames[last]) != fruit_count
    if partial_tail:
        del frames[last]
    if not frames or any(len(frame) != fruit_count for frame in frames.values()):
        raise ValueError("missing fruit inside the complete frame prefix")
    last = max(frames)
    previous_time = -1.0
    for frame, fruits in sorted(frames.items()):
        times = {fruit["time_s"] for fruit in fruits.values()}
        if len(times) != 1:
            raise ValueError("fruits in one frame have different times")
        time = times.pop()
        if time <= previous_time or (frame == 0 and time != 0):
            raise ValueError("frame times must increase from zero")
        previous_time = time

    summaries = []
    minimum_clearance = math.inf
    for index in range(fruit_count):
        history = [fruits[index] for _, fruits in sorted(frames.items())]
        radius = history[0]["radius_m"]
        released = False
        release_frame = landing_frame = None
        maximum_descent_speed = 0.0
        for frame, state in enumerate(history):
            if state["radius_m"] != radius:
                raise ValueError("fruit radius changed within the trajectory")
            if released and not state["released"]:
                raise ValueError("latched release bit was cleared")
            clearance = state["z_m"] - radius
            minimum_clearance = min(minimum_clearance, clearance)
            if state["released"]:
                if not released:
                    release_frame = frame
                released = True
                maximum_descent_speed = max(maximum_descent_speed, -state["vz_m_s"])
                if (landing_frame is None and abs(clearance) <= 2e-6 and
                        abs(state["vz_m_s"]) <= 1e-3):
                    landing_frame = frame
        final = history[-1]
        summaries.append({
            "fruit": index,
            "first_release_frame": release_frame,
            "first_supported_frame_after_release": landing_frame,
            "maximum_downward_speed_after_release_m_s": maximum_descent_speed,
            "final_clearance_m": final["z_m"] - radius,
            "final_vertical_velocity_m_s": final["vz_m_s"],
            "final_speed_m_s": math.sqrt(sum(final[key] ** 2 for key in ("vx_m_s", "vy_m_s", "vz_m_s"))),
            "final_last_substep_support_impulse_Ns": final["last_substep_ground_impulse_Ns"],
        })
    released_fruits = [row["fruit"] for row in summaries if row["first_release_frame"] is not None]
    never_supported = [row["fruit"] for row in summaries if
                       row["first_release_frame"] is not None and
                       row["first_supported_frame_after_release"] is None]
    return {
        "trace": str(path),
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "last_complete_frame": last,
        "simulated_seconds": previous_time,
        "expected_frames": expected_frames,
        "complete": last == expected_frames,
        "incomplete_tail_ignored": partial_tail,
        "minimum_fruit_clearance_m": minimum_clearance,
        "ground_penetration_free": minimum_clearance >= -1e-6,
        "released_fruits": released_fruits,
        "released_fruits_never_supported": never_supported,
        "fruits": summaries,
        "evidence_boundary": "Support is an observed radius/velocity match. Release is latched and can include re-entry. This report does not establish free-flight force balance, cloth contact residuals, calibrated restitution, or solver replay.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-frames", type=int, default=480)
    parser.add_argument("--fruit-count", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_frames < 1 or args.fruit_count < 1:
        parser.error("expected frames and fruit count must be positive")
    try:
        result = audit(args.trace, args.expected_frames, args.fruit_count)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0 if result["complete"] and result["ground_penetration_free"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
