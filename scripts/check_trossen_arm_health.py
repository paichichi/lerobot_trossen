#!/usr/bin/env python3
"""Read-only WidowX AI controller diagnostics; never sends an arm command."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import trossen_arm


def plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: plain(item) for key, item in vars(value).items()}
    return repr(value)


def finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def inspect_arm(name: str, ip_address: str, end_effector: Any) -> dict[str, Any]:
    driver = trossen_arm.TrossenArmDriver()
    try:
        # clear_error=False is important: this diagnostic must not mutate the
        # controller state or hide a fault that the operator needs to see.
        driver.configure(
            model=trossen_arm.Model.wxai_v0,
            end_effector=end_effector,
            serv_ip=ip_address,
            clear_error=False,
            timeout=3.0,
        )
        positions = list(driver.get_all_positions())
        velocities = list(driver.get_all_velocities())
        efforts = list(driver.get_all_efforts())
        external_efforts = list(driver.get_all_external_efforts())
        driver_temperatures = list(driver.get_all_driver_temperatures())
        rotor_temperatures = list(driver.get_all_rotor_temperatures())
        return {
            "name": name,
            "ip_address": ip_address,
            "connected": bool(driver.get_is_configured()),
            "driver_version": plain(driver.get_driver_version()),
            "controller_version": plain(driver.get_controller_version()),
            "num_joints": int(driver.get_num_joints()),
            "error_information": plain(driver.get_error_information()),
            "modes": plain(driver.get_modes()),
            "positions_rad": positions,
            "velocities_rad_s": velocities,
            "efforts": efforts,
            "external_efforts": external_efforts,
            "driver_temperatures_c": driver_temperatures,
            "rotor_temperatures_c": rotor_temperatures,
            "all_numeric_values_finite": all(
                finite(values)
                for values in (
                    positions,
                    velocities,
                    efforts,
                    external_efforts,
                    driver_temperatures,
                    rotor_temperatures,
                )
            ),
            "max_abs_velocity_rad_s": max(map(abs, velocities), default=0.0),
            "max_driver_temperature_c": max(driver_temperatures, default=float("nan")),
            "max_rotor_temperature_c": max(rotor_temperatures, default=float("nan")),
        }
    finally:
        if driver.get_is_configured():
            driver.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-ip", default="192.168.1.4")
    parser.add_argument("--leader-ip", default="192.168.1.2")
    args = parser.parse_args()

    reports = []
    failures = []
    arms = (
        ("follower", args.follower_ip, trossen_arm.StandardEndEffector.wxai_v0_follower),
        ("leader", args.leader_ip, trossen_arm.StandardEndEffector.wxai_v0_leader),
    )
    for name, ip_address, end_effector in arms:
        try:
            reports.append(inspect_arm(name, ip_address, end_effector))
        except Exception as exc:  # Keep checking the other arm if one is offline.
            failures.append({"name": name, "ip_address": ip_address, "error": repr(exc)})

    print(json.dumps({"arms": reports, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
