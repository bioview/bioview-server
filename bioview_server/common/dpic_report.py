"""Shared logging and serialisation for DPIC balance results.

Both the USRP and dummy backends run the same balancer, so they report it the
same way. Every outcome -- including "it did not run" -- carries a reason: a
balance that bails out must never reach the client as a success.
"""

from __future__ import annotations

from bioview_common import log_print
from bioview_common.signal_schemes.dpic import DpicBalancer


#: ``dpic_balance`` config key -> ``DpicBalancer`` field. Every knob the VI
#: exposes is here, so a rig can be re-timed from the config file without a
#: code change; the defaults are the VI's own numbers.
BALANCER_KEYS = (
    "coarse_phase_step_deg",
    "coarse_amp_step",
    "coarse_probe_amplitude",
    "phase_step_deg",
    "amp_step",
    "max_amplitude",
    "time_budget_s",
    "coarse_settle_time_s",
    "fine_settle_time_s",
    "stage_settle_time_s",
    "amp_target",
    "amp_tolerance",
    "gain_step_db",
    "gain_settle_time_s",
    "max_gain_steps",
)


def build_balancer(dpic_cfg: dict | None, **hooks) -> DpicBalancer:
    """Build a balancer from a group's ``dpic_balance`` block.

    Both RF backends run the same search, so they configure it the same way;
    a key the config omits keeps the balancer's own default rather than a
    default repeated at each call site and free to drift.
    """
    dpic_cfg = dpic_cfg or {}
    settings = {
        key: dpic_cfg[key] for key in BALANCER_KEYS if dpic_cfg.get(key) is not None
    }
    return DpicBalancer(**settings, **hooks)


def balance_outcome(logger, results) -> dict:
    """Log each pair's result and summarise the run.

    Returns ``{"ok": bool, "message": str, "results": [...]}``. ``ok`` is False
    unless every pair converged, so a partial balance is reported as a failure
    with the pairs that failed named in the message.
    """
    failures = []

    for r in results:
        where = f"Tx{r.inject_tx}->Tx{r.measure_tx}/Rx{r.measure_rx}"

        for stage in r.stages:
            log_print(logger, "debug", f"[DPIC] {where} {stage.describe()}")

        if not r.converged:
            reason = r.message or "no metric was readable"
            failures.append(f"{where}: {reason}")
            log_print(
                logger,
                "error",
                f"[DPIC] {where}: {reason}; previous phase/amplitude restored.",
            )
            continue

        log_print(
            logger,
            "info",
            f"[DPIC] {where} [{r.method}]: phase={r.best_phase_deg:.2f} deg "
            f"amp={r.best_amplitude:.3f} gain={r.inject_gain_db:.1f} dB "
            f"null={r.null_depth_db:.1f} dB "
            f"({r.num_measurements} reads in {r.elapsed_s:.1f} s)",
        )
        if r.truncated:
            cut = [s.name for s in r.stages if s.truncated]
            log_print(
                logger,
                "warning",
                f"[DPIC] {where}: time budget expired during {', '.join(cut)}. "
                "The result is the best point seen, not a completed search -- "
                "raise dpic_balance.time_budget_s.",
            )

    if not results:
        return {"ok": False, "message": "No DPIC pairs to balance.", "results": []}

    payload = {
        "ok": not failures,
        "message": (
            "; ".join(failures)
            if failures
            else f"DPIC balance complete for {len(results)} pair(s)."
        ),
        "results": serialize(results),
    }
    return payload


def serialize(results) -> list[dict]:
    """Result fields the client and the saved config keep."""
    return [
        {
            "inject_tx": r.inject_tx,
            "measure_tx": r.measure_tx,
            "measure_rx": r.measure_rx,
            "best_phase_deg": r.best_phase_deg,
            "best_amplitude": r.best_amplitude,
            "inject_gain_db": r.inject_gain_db,
            "min_metric": r.min_metric,
            "start_metric": r.start_metric,
            "null_depth_db": r.null_depth_db,
            "method": r.method,
            "elapsed_s": r.elapsed_s,
            "num_measurements": r.num_measurements,
            "converged": r.converged,
            "truncated": r.truncated,
            "message": r.message,
            "stages": [
                {
                    "name": s.name,
                    "planned": s.planned,
                    "visited": s.visited,
                    "measured": s.measured,
                    "best_value": s.best_value,
                    "best_metric": s.best_metric,
                    "elapsed_s": s.elapsed_s,
                }
                for s in r.stages
            ],
        }
        for r in results
    ]
