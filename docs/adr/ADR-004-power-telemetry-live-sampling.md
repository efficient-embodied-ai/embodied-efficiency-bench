# ADR-004: Power via live sampling (`pynvml`, `tegrastats`, `amd-smi`); unmeasured Cloud TPU power reported as `UNAVAILABLE` (`null`)

- **Status:** Accepted: 2026-10-08; amended 2026-10-30 for v1.0 (`power_provenance = "UNAVAILABLE"` on Cloud TPU without physical rail sensor)
- **Deciders:** EEAI @ CoRL 2026 organizers (harness lead: maintainer; WS-A owner: EEAI Benchmark Committee)
- **Workstream / milestone:** WS-A · M1 (telemetry sidecar); TPU path WS-B · M2
- **Supersedes:** `embodied_bench.telemetry.TelemetryMonitor.measure_operating_point` TDP-table power model (parsers real, sampling simulated; see `docs/CURRENT_STATE.md`)

## Context

Two of the four leaderboard views ("per-Watt" and, indirectly, "per-$") depend on energy per action / per frame. Vendors expose power very differently: NVIDIA discrete GPUs via NVML (`nvmlDeviceGetPowerUsage` and monotonic `nvmlDeviceGetTotalEnergyConsumption` in `energy_mj`, ~100 ms update), Jetson SoCs via `tegrastats` INA3221 rails (`VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`), AMD via `amd-smi` / `rocm-smi` (socket/package power), Google Cloud TPU via `tpu-info` / `libtpu` (guest VM chip-level power telemetry is not exposed on public Cloud TPU v6e/v5e images). Prior to M1, legacy `telemetry.py` derived watts from TDP × a utilization guess and never sampled during the timed window.

## Decision

1. **A telemetry sidecar process samples at 10 Hz (`100 ms`) for the whole timed window** (plus an idle baseline before warm-up), recording `t, power_w, energy_mj, mem_used_mb, temp_c, sm_clock_mhz, source`. Backends: `pynvml` (NVIDIA dGPU), `tegrastats --interval 100` (Jetson), `amd-smi metric --power` (AMD), `tpu-info` / `torch.tpu` / `jax` memory telemetry (TPU).
2. **Energy per unit is integrated from measured samples (or monotonic NVML `energy_mj` counters), never silently multiplied from TDP:** `J_unit = ∫ P dt over the window / units_completed`, reported alongside `mean_power_w`, `idle_power_w`, and `peak_mem_used_gb`.
3. **Provenance is mandatory:** each report carries `power_provenance ∈ {"MEASURED", "ESTIMATED", "UNAVAILABLE"}` plus `power_source`. **Cloud TPU VMs without physical guest power counters report `power_provenance = "UNAVAILABLE"` with `mean_power_watts = null` and `joules_per_unit = null` by default** (they are omitted from the Secondary Perf-per-Watt toggle rather than fabricated from TDP). An operator may explicitly pass `--assume-power-watts <W>`, which tags the result `power_provenance = "ESTIMATED"`, renders with a distinct marker, and remains ineligible for per-Watt awards.
4. **System vs. device power:** Jetson `VDD_IN` is whole-module power; dGPU readings are board power. Reports record `power_source` and scope explicitly; per-Watt views state which sensor rail they use.
5. Peak memory is read from the same sidecar (`nvmlDeviceGetMemoryInfo`, `tegrastats RAM`, `amd-smi`, `torch.tpu` / `tpu-info` HBM) and feeds the single-chip envelope check (`≤ 64 GB`).

## Consequences

**Positive**
- Honest measured per-Watt numbers on NVIDIA/AMD/Jetson from day one; Cloud TPU rows remain first-class on raw throughput, per-100-BF16-TFLOPS, and per-$1k-CapEx views without fabricated wattage.
- The sidecar is independent of the participant's process, so it cannot be gamed by the model server.

**Negative / costs**
- 10 Hz NVML/tegrastats sampling has ±1–2 % overhead on small SoCs; measured and documented, not corrected for.
- `pynvml` power on some consumer boards reports board power with coarse update cadence; monotonic `nvmlDeviceGetTotalEnergyConsumption` (`energy_mj`) and sustained multi-hundred-call windows mitigate discretization noise.
- Requires root/`sudo` for `tegrastats` rails on some JetPack versions; documented in the Jetson image.

**Risks**
- `tpu-info` output format changes across `libtpu` versions → parser falls back cleanly to `torch.tpu` / `jax` HBM queries and reports `power_provenance = "UNAVAILABLE"` rather than failing the run or inventing wattage.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Keep TDP-table estimates for all vendors | Not a measurement; identical "power" for every submission on a given SKU makes per-Watt meaningless. |
| Silently fabricate TPU watts from nameplate TDP | Violates strict hardware telemetry integrity; `UNAVAILABLE` (`null`) is strictly more honest unless explicitly requested via `--assume-power-watts` (`ESTIMATED`). |
| External power meter (e.g., Yokogawa / smart PDU) as the only source | Gold standard but unavailable to most participants and cloud instances; kept as an optional organizer cross-check for Jetson. |
| `nvidia-smi` polling via subprocess | ~50 ms per call and process-spawn jitter; `pynvml` exposes the same counters in-process without spawn overhead. |

## References

- `ENGINEERING_PLAN.md` §2.1, §4 WS-A, §6 hardware
- NVML API (`nvmlDeviceGetPowerUsage`, `nvmlDeviceGetTotalEnergyConsumption`), Jetson `tegrastats` docs, AMD SMI, `tpu-info`
