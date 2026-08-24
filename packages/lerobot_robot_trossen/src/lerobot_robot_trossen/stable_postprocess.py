from __future__ import annotations


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def home_tracking_errors(
    target: list[float], observed: list[float]
) -> tuple[float, float]:
    """Return peak arm and gripper errors for a seven-joint home pose."""
    if len(target) != 7 or len(observed) != 7:
        raise ValueError("Home target and observation must each contain seven values")
    arm_error = max(
        abs(actual - expected)
        for actual, expected in zip(observed[:6], target[:6], strict=True)
    )
    gripper_error = abs(observed[6] - target[6])
    return arm_error, gripper_error


class TimeAwareJointTargetFilter:
    """Rate-limit absolute joint targets without changing their representation."""

    def __init__(
        self,
        *,
        arm_joint_names: tuple[str, ...],
        gripper_joint_name: str,
        nominal_dt: float,
        max_dt_multiplier: float,
        arm_max_velocity_rad_s: float,
        arm_max_acceleration_rad_s2: float,
        gripper_max_velocity_m_s: float,
        gripper_deadband_m: float,
    ) -> None:
        self.arm_joint_names = arm_joint_names
        self.gripper_joint_name = gripper_joint_name
        self.nominal_dt = nominal_dt
        self.max_dt = nominal_dt * max_dt_multiplier
        self.arm_max_velocity = arm_max_velocity_rad_s
        self.arm_max_acceleration = arm_max_acceleration_rad_s2
        self.gripper_max_velocity = gripper_max_velocity_m_s
        self.gripper_deadband = gripper_deadband_m
        self._last_time: float | None = None
        self._arm_velocities = dict.fromkeys(arm_joint_names, 0.0)

    def reset(self) -> None:
        self._last_time = None
        self._arm_velocities = dict.fromkeys(self.arm_joint_names, 0.0)

    def _step_dt(self, now: float) -> float:
        if self._last_time is None:
            dt = self.nominal_dt
        else:
            elapsed = now - self._last_time
            if elapsed > max(0.5, 4.0 * self.nominal_dt):
                self._arm_velocities = dict.fromkeys(self.arm_joint_names, 0.0)
                dt = self.nominal_dt
            else:
                dt = _clip(elapsed, 1e-4, self.max_dt)
        self._last_time = now
        return dt

    def apply(
        self,
        desired_positions: dict[str, float],
        present_positions: dict[str, float],
        *,
        now: float,
    ) -> dict[str, float]:
        dt = self._step_dt(now)
        filtered = {name: float(value) for name, value in desired_positions.items()}

        for name in self.arm_joint_names:
            if name not in filtered:
                continue
            present = float(present_positions[name])
            error = filtered[name] - present
            desired_velocity = _clip(
                error / dt,
                -self.arm_max_velocity,
                self.arm_max_velocity,
            )
            previous_velocity = self._arm_velocities[name]
            max_velocity_change = self.arm_max_acceleration * dt
            velocity = _clip(
                desired_velocity,
                previous_velocity - max_velocity_change,
                previous_velocity + max_velocity_change,
            )
            step = velocity * dt
            if abs(step) > abs(error):
                step = error
                velocity = step / dt
            filtered[name] = present + step
            self._arm_velocities[name] = velocity

        name = self.gripper_joint_name
        if name in filtered:
            present = float(present_positions[name])
            error = filtered[name] - present
            if abs(error) <= self.gripper_deadband:
                filtered[name] = present
            else:
                max_step = self.gripper_max_velocity * dt
                filtered[name] = present + _clip(error, -max_step, max_step)

        return filtered
