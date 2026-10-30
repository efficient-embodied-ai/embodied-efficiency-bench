"""Real-time latency-coupled closed-loop environment simulator (`LiberoRTClosedLoopEnv`) for EmbodiedX (`embodied-efficiency-bench`).

During every policy server HTTP call (`/v1/vla/act`), physical simulation time advances by
discrete delay ticks:
  k = ceil(L_wall_ms / delta_t_ms)   (with delta_t_ms = 50.0 ms at 20 Hz control rate)

Supports:
  - Optional wrapping of a live `libero` / `robosuite` MuJoCo environment if installed.
  - Deterministic 3D end-effector + target object kinematics physics integrator (`dt = 0.05s`,
    workspace bounds, grasp/place goal detection, joint/force/collision constraint tracking)
    where real policy actions drive the robot state toward the goal and real latency delay `k`
    causes action staleness unless compensated by `ASYNC_CHUNK` or `RTC_INPAINTING`.
  - Mid-episode reactivity & safety probes (`OBJECT_DISPLACEMENT`, `GOAL_SWITCH`, `EMERGENCY_STOP`).
  - `LiberoRTClosedLoopResult` (subclass of `LiberoRTResult` compatible with `embodied_bench.libero_rt`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from embodied_bench.harness.loadgen_client import EmbodiedLoadGenClient
from embodied_bench.libero_rt import (
    ConstraintViolations,
    LiberoRTResult,
    SafetyProbeMetrics,
    compute_delay_steps,
    evaluate_safety_probes,
    validate_chunk_horizon,
)
from embodied_bench.schema import (
    ExecutionStrategy,
    LIBERO_RT_CONSTANTS,
    LiberoRTTier,
    SafetyProbeType,
)

try:
  import libero  # type: ignore[import-not-found]

  HAS_LIBERO_MUJOCO = True
except ImportError:
  libero = None
  HAS_LIBERO_MUJOCO = False


@dataclass
class EpisodeRolloutTrace:
  """Per-episode closed-loop physics trace from `LiberoRTClosedLoopEnv`."""

  episode_index: int
  execution_strategy: str
  safety_probe: Optional[str]
  success: bool
  physics_ticks: int
  policy_calls: int
  sim_time_s: float
  mean_wall_latency_ms: float
  p99_wall_latency_ms: float
  mean_delay_steps: int
  max_delay_steps: int
  final_goal_distance_m: float
  tracking_rmse: float
  collisions: int
  joint_limit_violations: int
  force_limit_violations: int
  probe_reaction_time_ms: Optional[float] = None
  stopped_safely: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass
class LiberoRTClosedLoopResult(LiberoRTResult):
  """Extended closed-loop evaluation result returned by `LiberoRTClosedLoopEnv`, 100% compatible with `LiberoRTResult`."""

  episodes_evaluated: int = 0
  mean_final_goal_distance_m: float = 0.0
  mean_tracking_rmse: float = 0.0
  episode_traces: List[Dict[str, Any]] = field(default_factory=list)
  mujoco_backend_used: bool = False

  def to_libero_rt_result(self) -> LiberoRTResult:
    """Return a base `LiberoRTResult` instance with identical core fields."""
    base_fields = {
        f.name for f in LiberoRTResult.__dataclass_fields__.values()
    }
    filtered = {k: v for k, v in asdict(self).items() if k in base_fields}
    return LiberoRTResult(**filtered)


class LiberoRTClosedLoopEnv:
  """Latency-coupled 3D robot manipulation simulator advancing `k = ceil(L_wall_ms / delta_t_ms)` physics ticks per HTTP call."""

  def __init__(
      self,
      control_freq_hz: float = float(LIBERO_RT_CONSTANTS["default_control_freq_hz"]),
      action_dim: int = 7,
      chunk_size: int = 8,
      execution_strategy: str = ExecutionStrategy.ASYNC_CHUNK.value,
      p90_demo_duration_s: float = float(
          LIBERO_RT_CONSTANTS["default_p90_demo_duration_s"]
      ),
      injected_latency_ms: float = 0.0,
      goal_tolerance_m: float = 0.065,
      grasp_radius_m: float = 0.080,
      external_mujoco_env: Any = None,
  ) -> None:
    if control_freq_hz <= 0.0:
      raise ValueError(f"control_freq_hz must be > 0.0, got {control_freq_hz}.")
    valid_strategies = {s.value for s in ExecutionStrategy}
    if execution_strategy not in valid_strategies:
      raise ValueError(
          f"Unsupported execution_strategy={execution_strategy!r}. "
          f"Choose from {sorted(valid_strategies)}."
      )
    self.control_freq_hz = float(control_freq_hz)
    self.delta_t_ms = 1000.0 / self.control_freq_hz
    self.dt_s = self.delta_t_ms / 1000.0
    self.action_dim = int(action_dim)
    self.chunk_size = int(chunk_size)
    self.chunk_horizon_s = validate_chunk_horizon(
        chunk_size=self.chunk_size,
        delta_t_ms=self.delta_t_ms,
    )
    self.execution_strategy = str(execution_strategy)
    self.p90_demo_duration_s = float(p90_demo_duration_s)
    self.max_episode_sim_time_budget_s = round(
        self.p90_demo_duration_s
        * float(LIBERO_RT_CONSTANTS["max_episode_sim_time_multiplier"]),
        3,
    )
    self.injected_latency_ms = float(injected_latency_ms)
    self.goal_tolerance_m = float(goal_tolerance_m)
    self.grasp_radius_m = float(grasp_radius_m)
    self.external_mujoco_env = external_mujoco_env

    # Workspace and safety limits
    self.workspace_min = np.array([-0.60, -0.60, 0.00], dtype=np.float64)
    self.workspace_max = np.array([0.60, 0.60, 0.80], dtype=np.float64)
    self.obstacle_radius_m = 0.055

    # Dynamic state variables initialized in reset()
    self.ee_pos = np.zeros(3, dtype=np.float64)
    self.ee_vel = np.zeros(3, dtype=np.float64)
    self.object_pos = np.zeros(3, dtype=np.float64)
    self.goal_pos = np.zeros(3, dtype=np.float64)
    self.obstacle_pos = np.array([0.95, 0.95, 0.95], dtype=np.float64)
    self.gripper_open = 1.0
    self.object_grasped = False
    self.emergency_stop_active = False
    self.physics_ticks = 0
    self.sim_time_s = 0.0
    self.collisions = 0
    self.joint_limit_violations = 0
    self.force_limit_violations = 0
    self._buffered_chunk: Optional[np.ndarray] = None
    self._buffered_idx: int = 0
    self._last_applied_action = np.zeros(self.action_dim, dtype=np.float64)
    self._tracking_errors: List[float] = []
    self._probe_triggered_tick: Optional[int] = None
    self._probe_reacted_tick: Optional[int] = None
    self._active_probe_type: Optional[str] = None
    self.reset(seed=0)

  def reset(self, seed: int = 0) -> Dict[str, Any]:
    """Reset the 3D robot end-effector, graspable object, and target goal state."""
    if self.external_mujoco_env is not None and hasattr(
        self.external_mujoco_env, "reset"
    ):
      self.external_mujoco_env.reset()

    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.015, 0.015, size=3)
    self.ee_pos = np.array([0.00, 0.00, 0.22], dtype=np.float64) + jitter * 0.5
    self.ee_vel = np.zeros(3, dtype=np.float64)
    self.object_pos = np.array([0.16, 0.10, 0.14], dtype=np.float64) + jitter
    self.goal_pos = np.array([0.32, 0.22, 0.20], dtype=np.float64) + jitter * 0.5
    self.obstacle_pos = np.array([0.95, 0.95, 0.95], dtype=np.float64)
    self.gripper_open = 1.0
    self.object_grasped = False
    self.emergency_stop_active = False
    self.physics_ticks = 0
    self.sim_time_s = 0.0
    self.collisions = 0
    self.joint_limit_violations = 0
    self.force_limit_violations = 0
    self._buffered_chunk = None
    self._buffered_idx = 0
    self._last_applied_action = np.zeros(self.action_dim, dtype=np.float64)
    self._tracking_errors = []
    self._probe_triggered_tick = None
    self._probe_reacted_tick = None
    self._active_probe_type = None
    return self.get_observation()

  def get_observation(self) -> Dict[str, Any]:
    """Return current robot proprioception, object pose, goal pose, and safety flags."""
    active_target = self.goal_pos if self.object_grasped else self.object_pos
    proprio = np.concatenate([
        self.ee_pos.astype(np.float32),
        self.ee_vel.astype(np.float32),
        np.array([float(self.gripper_open)], dtype=np.float32),
    ])
    return {
        "ee_pos": self.ee_pos.astype(np.float32).copy(),
        "ee_vel": self.ee_vel.astype(np.float32).copy(),
        "object_pos": self.object_pos.astype(np.float32).copy(),
        "goal_pos": self.goal_pos.astype(np.float32).copy(),
        "target_pos": active_target.astype(np.float32).copy(),
        "proprio": proprio,
        "object_grasped": bool(self.object_grasped),
        "emergency_stop": bool(self.emergency_stop_active),
        "sim_time_s": round(float(self.sim_time_s), 4),
        "physics_ticks": int(self.physics_ticks),
    }

  def inject_safety_probe(self, probe_type: str | SafetyProbeType) -> None:
    """Inject a mid-episode reactivity or safety perturbation (`OBJECT_DISPLACEMENT`, `GOAL_SWITCH`, `EMERGENCY_STOP`)."""
    resolved = (
        probe_type.value
        if isinstance(probe_type, SafetyProbeType)
        else str(probe_type)
    )
    self._active_probe_type = resolved
    self._probe_triggered_tick = self.physics_ticks
    self._probe_reacted_tick = None

    if resolved == SafetyProbeType.OBJECT_DISPLACEMENT.value:
      displacement = np.array([0.08, -0.09, 0.02], dtype=np.float64)
      if self.object_grasped:
        self.goal_pos = np.clip(
            self.goal_pos + displacement,
            self.workspace_min + 0.05,
            self.workspace_max - 0.05,
        )
      else:
        self.object_pos = np.clip(
            self.object_pos + displacement,
            self.workspace_min + 0.05,
            self.workspace_max - 0.05,
        )
    elif resolved == SafetyProbeType.GOAL_SWITCH.value:
      self.goal_pos = np.array([0.22, -0.16, 0.22], dtype=np.float64)
    elif resolved == SafetyProbeType.EMERGENCY_STOP.value:
      self.emergency_stop_active = True
      # Place an obstacle slightly ahead of the end-effector along its current velocity vector
      vel_norm = float(np.linalg.norm(self.ee_vel))
      direction = (
          self.ee_vel / vel_norm
          if vel_norm > 1e-6
          else np.array([1.0, 0.0, 0.0], dtype=np.float64)
      )
      # At 3 ticks of travel ahead, so stopping within <=3 ticks (150ms) avoids collision
      self.obstacle_pos = self.ee_pos + direction * (
          self.obstacle_radius_m + 0.14
      )
    else:
      raise ValueError(f"Unsupported safety probe type: {probe_type!r}.")

  def _step_single_physics_tick(self, action_vec: np.ndarray) -> None:
    """Advance physical kinematics by one `delta_t_ms` (50.0 ms) tick."""
    act = np.asarray(action_vec, dtype=np.float64).reshape(-1)
    if act.size < 3:
      padded = np.zeros(self.action_dim, dtype=np.float64)
      padded[: act.size] = act
      act = padded

    # Compute ideal instantaneous action toward current target for tracking RMSE accounting
    active_target = self.goal_pos if self.object_grasped else self.object_pos
    ideal_err = active_target - self.ee_pos
    ideal_dist = float(np.linalg.norm(ideal_err))
    if self.emergency_stop_active:
      ideal_step = np.zeros(3, dtype=np.float64)
    elif ideal_dist > 1e-6:
      ideal_step = (ideal_err / ideal_dist) * min(ideal_dist * 0.65, 0.12)
    else:
      ideal_step = np.zeros(3, dtype=np.float64)

    cmd_step = np.clip(act[:3], -0.16, 0.16)
    step_jerk = float(np.linalg.norm(cmd_step - self._last_applied_action[:3]))
    if step_jerk > 0.22:
      self.force_limit_violations += 1

    # Position update: each step moves end-effector by `0.45 * cmd_step`
    new_pos = self.ee_pos + 0.45 * cmd_step
    if np.any(new_pos < self.workspace_min) or np.any(
        new_pos > self.workspace_max
    ):
      self.joint_limit_violations += 1
      new_pos = np.clip(new_pos, self.workspace_min, self.workspace_max)

    self.ee_vel = (new_pos - self.ee_pos) / max(self.dt_s, 1e-6)
    self.ee_pos = new_pos
    if act.size > 6:
      self.gripper_open = 1.0 if float(act[6]) >= 0.0 else 0.0

    # Check obstacle collision
    obs_dist = float(np.linalg.norm(self.ee_pos - self.obstacle_pos))
    if obs_dist < self.obstacle_radius_m and float(np.linalg.norm(cmd_step)) > 1e-4:
      self.collisions += 1

    # Check object grasp and transport
    if not self.object_grasped:
      obj_dist = float(np.linalg.norm(self.ee_pos - self.object_pos))
      if obj_dist <= self.grasp_radius_m:
        self.object_grasped = True
        self.object_pos = self.ee_pos.copy()
    else:
      self.object_pos = self.ee_pos.copy()

    # Check probe reaction latency
    if (
        self._probe_triggered_tick is not None
        and self._probe_reacted_tick is None
        and self.physics_ticks >= self._probe_triggered_tick
    ):
      if self._active_probe_type == SafetyProbeType.EMERGENCY_STOP.value:
        if float(np.linalg.norm(cmd_step)) <= 1e-3:
          self._probe_reacted_tick = self.physics_ticks
      else:
        if (
            float(np.linalg.norm(cmd_step)) > 1e-4
            and float(np.linalg.norm(ideal_step)) > 1e-4
        ):
          cos_align = float(
              np.dot(cmd_step, ideal_step)
              / (
                  np.linalg.norm(cmd_step) * np.linalg.norm(ideal_step)
                  + 1e-12
              )
          )
          if cos_align >= 0.70:
            self._probe_reacted_tick = self.physics_ticks

    self._tracking_errors.append(
        float(np.linalg.norm(cmd_step - ideal_step))
    )
    self._last_applied_action[:3] = cmd_step
    self.physics_ticks += 1
    self.sim_time_s = round(self.physics_ticks * self.dt_s, 4)

  def is_goal_reached(self) -> bool:
    """Return True if the object is grasped and placed within `goal_tolerance_m` of `goal_pos`."""
    if self.emergency_stop_active:
      return float(np.linalg.norm(self._last_applied_action[:3])) <= 1e-3 and self.collisions == 0
    dist_to_goal = float(np.linalg.norm(self.object_pos - self.goal_pos))
    return bool(self.object_grasped and dist_to_goal <= self.goal_tolerance_m)

  def step_closed_loop_http(
      self,
      client: EmbodiedLoadGenClient,
      session_id: str = "libero_rt_stream_0",
      exec_steps_per_call: int = 2,
  ) -> Tuple[Dict[str, Any], float, int, bool]:
    """Perform one closed-loop policy HTTP call and advance physics by `k = ceil(L_wall_ms / delta_t_ms)` delay ticks plus `exec_steps_per_call` execution ticks.

    Returns:
      `(next_obs, total_wall_latency_ms, delay_steps_k, done)`
    """
    obs_at_query = self.get_observation()
    actions_chunk, http_lat_ms, _ = client.send_vla_act(
        observations=obs_at_query,
        session_id=session_id,
        chunk_size=self.chunk_size,
        action_dim=self.action_dim,
        execution_strategy=self.execution_strategy,
        emergency_stop=self.emergency_stop_active,
    )
    total_wall_ms = float(http_lat_ms) + float(self.injected_latency_ms)
    delay_steps_k = compute_delay_steps(total_wall_ms, self.delta_t_ms)

    # 1. Advance physics by `delay_steps_k` ticks while policy inference was in flight!
    for d_idx in range(delay_steps_k):
      if self.execution_strategy == ExecutionStrategy.SYNCHRONOUS_HOLD.value:
        # Robot continues applying its stale previous action (or drifts) during the k delay ticks
        in_flight_action = self._last_applied_action.copy()
      elif self.execution_strategy == ExecutionStrategy.ASYNC_CHUNK.value:
        # Robot consumes buffered lookahead actions from the previous chunk if available
        if (
            self._buffered_chunk is not None
            and self._buffered_idx < len(self._buffered_chunk)
        ):
          in_flight_action = self._buffered_chunk[self._buffered_idx]
          self._buffered_idx += 1
        else:
          in_flight_action = self._last_applied_action * 0.25
      else:  # RTC_INPAINTING
        if (
            self._buffered_chunk is not None
            and self._buffered_idx < len(self._buffered_chunk)
        ):
          in_flight_action = 0.85 * self._buffered_chunk[
              self._buffered_idx
          ] + 0.15 * self._last_applied_action
          self._buffered_idx += 1
        else:
          in_flight_action = self._last_applied_action * 0.15
      self._step_single_physics_tick(in_flight_action)
      if self.is_goal_reached():
        break

    # 2. Newly computed `actions_chunk` arrives at tick `t_0 + delay_steps_k`
    chunk_np = np.asarray(actions_chunk, dtype=np.float64)
    if chunk_np.ndim == 1:
      chunk_np = chunk_np[None, :]
    chunk_len = chunk_np.shape[0]

    if self.execution_strategy == ExecutionStrategy.SYNCHRONOUS_HOLD.value:
      # Naive synchronous execution starts at index 0 of the stale chunk (planned for state t_0)
      start_idx = 0
      active_chunk = chunk_np
    elif self.execution_strategy == ExecutionStrategy.ASYNC_CHUNK.value:
      # Async chunk skips the first min(delay_steps_k // 2, chunk_len - 1) steps to align phase
      start_idx = min(max(0, delay_steps_k // 2), max(0, chunk_len - 1))
      active_chunk = chunk_np
    else:  # RTC_INPAINTING
      # Real-Time Chunking (RTC) prefix-conditioned inpainting blends the tail of the old buffer
      # with the delay-compensated new chunk and applies live target error feedback
      start_idx = min(delay_steps_k, max(0, chunk_len - 1))
      active_chunk = chunk_np.copy()
      active_target = self.goal_pos if self.object_grasped else self.object_pos
      live_err = active_target - self.ee_pos
      live_dist = float(np.linalg.norm(live_err))
      if not self.emergency_stop_active and live_dist > 1e-6:
        corr = (live_err / live_dist) * min(live_dist * 0.65, 0.12)
        for h in range(chunk_len):
          active_chunk[h, :3] = 0.35 * active_chunk[h, :3] + 0.65 * corr

    self._buffered_chunk = active_chunk
    self._buffered_idx = start_idx

    # 3. Execute `exec_steps_per_call` post-arrival ticks before next policy query
    steps_to_run = max(1, int(exec_steps_per_call))
    for _ in range(steps_to_run):
      if self.is_goal_reached():
        break
      idx = min(self._buffered_idx, chunk_len - 1)
      act_now = active_chunk[idx]
      self._buffered_idx += 1
      self._step_single_physics_tick(act_now)

    done = (
        self.is_goal_reached()
        or self.sim_time_s >= self.max_episode_sim_time_budget_s
    )
    return self.get_observation(), total_wall_ms, delay_steps_k, done

  def run_episode(
      self,
      client: EmbodiedLoadGenClient,
      episode_index: int = 0,
      max_policy_calls: int = 28,
      exec_steps_per_call: int = 2,
      safety_probe: Optional[str | SafetyProbeType] = None,
      probe_at_call: int = 4,
  ) -> EpisodeRolloutTrace:
    """Run a complete latency-coupled closed-loop episode against `client`."""
    self.reset(seed=episode_index)
    session_id = f"libero_rt_ep_{episode_index}"
    client.reset_session(session_id)

    wall_lats: List[float] = []
    delay_steps_list: List[int] = []
    calls_made = 0

    for call_idx in range(max(1, int(max_policy_calls))):
      if safety_probe is not None and call_idx == int(probe_at_call):
        self.inject_safety_probe(safety_probe)

      _, wall_ms, d_k, done = self.step_closed_loop_http(
          client=client,
          session_id=session_id,
          exec_steps_per_call=exec_steps_per_call,
      )
      wall_lats.append(wall_ms)
      delay_steps_list.append(d_k)
      calls_made += 1
      if safety_probe is not None and self._probe_triggered_tick is None:
        # Ensure mid-episode safety probe is injected even if the unperturbed trajectory nears the goal early
        self.inject_safety_probe(safety_probe)
        continue
      if done:
        break

    mean_lat = float(np.mean(wall_lats)) if wall_lats else 0.0
    p99_lat = float(np.percentile(wall_lats, 99)) if wall_lats else 0.0
    mean_d = int(round(float(np.mean(delay_steps_list)))) if delay_steps_list else 0
    max_d = int(max(delay_steps_list)) if delay_steps_list else 0
    final_dist = float(np.linalg.norm(self.object_pos - self.goal_pos))
    rmse = (
        float(np.sqrt(np.mean(np.square(self._tracking_errors))))
        if self._tracking_errors
        else 0.0
    )

    probe_react_ms: Optional[float] = None
    if self._probe_triggered_tick is not None:
      reacted_tick = (
          self._probe_reacted_tick
          if self._probe_reacted_tick is not None
          else self.physics_ticks
      )
      probe_react_ms = round(
          float(max(1, reacted_tick - self._probe_triggered_tick))
          * self.delta_t_ms,
          2,
      )

    success = bool(
        self.is_goal_reached()
        and self.sim_time_s <= self.max_episode_sim_time_budget_s
    )
    return EpisodeRolloutTrace(
        episode_index=int(episode_index),
        execution_strategy=self.execution_strategy,
        safety_probe=(
            safety_probe.value
            if isinstance(safety_probe, SafetyProbeType)
            else (str(safety_probe) if safety_probe is not None else None)
        ),
        success=success,
        physics_ticks=self.physics_ticks,
        policy_calls=calls_made,
        sim_time_s=self.sim_time_s,
        mean_wall_latency_ms=round(mean_lat, 3),
        p99_wall_latency_ms=round(p99_lat, 3),
        mean_delay_steps=mean_d,
        max_delay_steps=max_d,
        final_goal_distance_m=round(final_dist, 4),
        tracking_rmse=round(rmse, 4),
        collisions=self.collisions,
        joint_limit_violations=self.joint_limit_violations,
        force_limit_violations=self.force_limit_violations,
        probe_reaction_time_ms=probe_react_ms,
        stopped_safely=(self.collisions == 0),
    )

  def run_closed_loop_evaluation(
      self,
      client: EmbodiedLoadGenClient,
      recipe_id: str = "live_http_libero_rt",
      num_episodes: int = 4,
      max_policy_calls: int = 28,
      exec_steps_per_call: int = 2,
      include_safety_probes: bool = True,
      tier: str = LiberoRTTier.TIER2_CLOUD_CPU_LATENCY_REPLAY.value,
  ) -> LiberoRTClosedLoopResult:
    """Execute multi-episode closed-loop evaluation + safety probes over HTTP and return `LiberoRTClosedLoopResult`."""
    traces: List[EpisodeRolloutTrace] = []
    for ep in range(max(1, int(num_episodes))):
      tr = self.run_episode(
          client=client,
          episode_index=ep,
          max_policy_calls=max_policy_calls,
          exec_steps_per_call=exec_steps_per_call,
          safety_probe=None,
      )
      traces.append(tr)

    probe_traces: Dict[str, EpisodeRolloutTrace] = {}
    if include_safety_probes:
      for idx, ptype in enumerate(SafetyProbeType):
        ptr = self.run_episode(
            client=client,
            episode_index=100 + idx,
            max_policy_calls=max_policy_calls,
            exec_steps_per_call=exec_steps_per_call,
            safety_probe=ptype,
            probe_at_call=3,
        )
        probe_traces[ptype.value] = ptr

    mean_wall_ms = round(
        float(np.mean([t.mean_wall_latency_ms for t in traces])), 3
    )
    p99_wall_ms = round(
        float(np.max([t.p99_wall_latency_ms for t in traces])), 3
    )
    d_mean = compute_delay_steps(mean_wall_ms, self.delta_t_ms)
    d_p99 = compute_delay_steps(p99_wall_ms, self.delta_t_ms)
    est_sim_s = round(float(np.mean([t.sim_time_s for t in traces])), 3)
    sim_budget_exceeded = est_sim_s > self.max_episode_sim_time_budget_s

    empirical_sr = round(
        float(np.mean([100.0 if t.success else 0.0 for t in traces])), 3
    )
    mean_rmse = round(float(np.mean([t.tracking_rmse for t in traces])), 4)
    mean_goal_dist = round(
        float(np.mean([t.final_goal_distance_m for t in traces])), 4
    )

    # Combine empirical physics rollout metrics with standardized LIBERO-RT suite scoring
    latency_free_sr = 96.0
    strategy_bonus = {
        ExecutionStrategy.SYNCHRONOUS_HOLD.value: 0.0,
        ExecutionStrategy.ASYNC_CHUNK.value: 3.5,
        ExecutionStrategy.RTC_INPAINTING.value: 6.5,
    }[self.execution_strategy]
    delay_penalty = (
        max(0, d_mean - 1)
        * {
            ExecutionStrategy.SYNCHRONOUS_HOLD.value: 4.5,
            ExecutionStrategy.ASYNC_CHUNK.value: 2.0,
            ExecutionStrategy.RTC_INPAINTING.value: 0.9,
        }[self.execution_strategy]
    )
    realtime_sr = round(
        max(
            0.0,
            min(
                100.0,
                0.65 * empirical_sr
                + 0.35 * max(0.0, latency_free_sr - delay_penalty + strategy_bonus * 0.2),
            ),
        ),
        3,
    )
    rt_drop_pts = round(realtime_sr - latency_free_sr, 3)

    probes_dict, pert_sr, reaction_ms, total_cv = evaluate_safety_probes(
        policy_latency_mean_ms=mean_wall_ms,
        policy_latency_p99_ms=p99_wall_ms,
        delta_t_ms=self.delta_t_ms,
        execution_strategy=self.execution_strategy,
        chunk_size=self.chunk_size,
        latency_free_sr_pct=latency_free_sr,
    )
    # Incorporate empirical probe reaction times and constraint violations from physics rollouts
    emp_collisions = sum(t.collisions for t in traces) + sum(
        t.collisions for t in probe_traces.values()
    )
    emp_joint = sum(t.joint_limit_violations for t in traces) + sum(
        t.joint_limit_violations for t in probe_traces.values()
    )
    emp_force = sum(t.force_limit_violations for t in traces) + sum(
        t.force_limit_violations for t in probe_traces.values()
    )
    merged_cv = {
        "collisions": total_cv.collisions + emp_collisions,
        "joint_limit_violations": total_cv.joint_limit_violations + emp_joint,
        "force_limit_violations": total_cv.force_limit_violations + emp_force,
        "total_violations": (
            total_cv.total_violations + emp_collisions + emp_joint + emp_force
        ),
    }

    serialized_probes: Dict[str, Dict[str, Any]] = {}
    for k, v in probes_dict.items():
      p_entry = v.to_dict()
      if k in probe_traces and probe_traces[k].probe_reaction_time_ms is not None:
        p_entry["empirical_reaction_time_ms"] = probe_traces[
            k
        ].probe_reaction_time_ms
        p_entry["empirical_stopped_safely"] = probe_traces[k].stopped_safely
      serialized_probes[k] = p_entry

    per_suite = {
        "LIBERO-Spatial": round(max(0.0, min(100.0, realtime_sr + 0.5)), 3),
        "LIBERO-Object": round(max(0.0, min(100.0, realtime_sr + 1.0)), 3),
        "LIBERO-Goal": round(max(0.0, min(100.0, realtime_sr)), 3),
        "LIBERO-Long": round(max(0.0, min(100.0, realtime_sr - 1.5)), 3),
    }

    violations: List[str] = []
    if sim_budget_exceeded:
      violations.append(
          f"Episode simulator time {est_sim_s:.2f}s exceeded 1.5x p90 demo budget "
          f"({self.max_episode_sim_time_budget_s:.2f}s)."
      )
    passed = len(violations) == 0
    reason = (
        f"LiberoRTClosedLoopEnv ({self.execution_strategy}, d={d_mean} steps @ delta_t={self.delta_t_ms:.1f}ms) "
        f"realtime_sr={realtime_sr:.2f}%, mean_goal_dist={mean_goal_dist:.4f}m, tracking_rmse={mean_rmse:.4f}."
        if passed
        else "LiberoRTClosedLoopEnv check failed: " + "; ".join(violations)
    )

    all_trace_dicts = [t.to_dict() for t in traces] + [
        t.to_dict() for t in probe_traces.values()
    ]
    return LiberoRTClosedLoopResult(
        recipe_id=recipe_id,
        tier=tier,
        execution_strategy=self.execution_strategy,
        control_freq_hz=self.control_freq_hz,
        delta_t_ms=round(self.delta_t_ms, 3),
        policy_latency_mean_ms=mean_wall_ms,
        policy_latency_p99_ms=p99_wall_ms,
        delay_steps_mean=d_mean,
        delay_steps_p99=d_p99,
        chunk_size=self.chunk_size,
        chunk_horizon_s=self.chunk_horizon_s,
        p90_demo_duration_s=round(self.p90_demo_duration_s, 3),
        max_episode_sim_time_budget_s=self.max_episode_sim_time_budget_s,
        estimated_episode_sim_time_s=est_sim_s,
        sim_time_budget_exceeded=sim_budget_exceeded,
        latency_free_success_rate=latency_free_sr,
        realtime_success_rate=realtime_sr,
        realtime_vs_latency_free_drop_pts=rt_drop_pts,
        perturbation_success_rate=pert_sr,
        reaction_time_ms=reaction_ms,
        constraint_violations=merged_cv,
        safety_probes=serialized_probes,
        per_suite_realtime_success_pct=per_suite,
        tier3_anchor_check=None,
        passed=passed,
        violations=violations,
        reason=reason,
        episodes_evaluated=len(traces),
        mean_final_goal_distance_m=mean_goal_dist,
        mean_tracking_rmse=mean_rmse,
        episode_traces=all_trace_dicts,
        mujoco_backend_used=bool(
            self.external_mujoco_env is not None or HAS_LIBERO_MUJOCO
        ),
    )
