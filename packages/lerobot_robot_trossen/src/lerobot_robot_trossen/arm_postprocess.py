from __future__ import annotations


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class TimeAwareArmTargetFilter:
    """Smooth absolute arm targets while leaving the gripper untouched."""

    def __init__(
        self,
        *,
        arm_joint_names: tuple[str, ...],
        nominal_dt: float,
        max_dt_multiplier: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> None:
        self.arm_joint_names = arm_joint_names
        self.nominal_dt = nominal_dt
        self.max_dt = nominal_dt * max_dt_multiplier
        self.max_velocity = max_velocity_rad_s
        self.max_acceleration = max_acceleration_rad_s2
        self._last_time: float | None = None
        self._velocities = dict.fromkeys(arm_joint_names, 0.0)

    def reset(self) -> None:
        self._last_time = None
        self._velocities = dict.fromkeys(self.arm_joint_names, 0.0)

    def _step_dt(self, now: float) -> float:
        if self._last_time is None:
            dt = self.nominal_dt
        else:
            elapsed = now - self._last_time
            if elapsed > max(0.5, 4.0 * self.nominal_dt):
                self._velocities = dict.fromkeys(self.arm_joint_names, 0.0)
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
                -self.max_velocity,
                self.max_velocity,
            )
            previous_velocity = self._velocities[name]
            max_velocity_change = self.max_acceleration * dt
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
            self._velocities[name] = velocity

        return filtered
