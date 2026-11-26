#!/usr/bin/env python3
"""Tune MoE cache settings by delegating fit and placement to llama-sweep-bench."""
import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

SHARES = ("0", "0.5", "0.95", "1")
MIN_GAIN_PERCENT = 5.0


def write_line(message, *, stream=sys.stdout, flush=False):
    stream.write(f"{message}\n")
    if flush:
        stream.flush()


CACHE_RE = re.compile(r"llama_moe_cache:\s*requested\s+([\d.]+)\s+MiB,\s*allocated\s+([\d.]+)\s+MiB")
CACHE_POLICY_RE = re.compile(r"impl:\s*active MoE cache policy:\s*([a-z0-9_-]+)", re.IGNORECASE)


def parse_rows(stderr):
    rows = []
    for line in stderr.decode(errors="replace").splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError
            n_kv, pp, tg = row["n_kv"], row["speed_pp"], row["speed_tg"]
            if isinstance(n_kv, bool) or not isinstance(n_kv, int):
                raise ValueError
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                   for v in (pp, tg)):
                raise ValueError
            rows.append(row)
        except (ValueError, KeyError, TypeError):
            raise ValueError("native JSONL is malformed or missing finite n_kv/speed fields") from None
    if len(rows) < 3:
        raise ValueError("native JSONL has fewer than three checkpoints")
    ids = [row["n_kv"] for row in rows]
    if ids != sorted(set(ids)):
        raise ValueError("native checkpoint IDs are duplicated or unordered")
    measured = rows[2:]
    scores = {key: math.exp(sum(math.log(row[key]) for row in measured) / len(measured))
              for key in ("speed_pp", "speed_tg")}
    return scores, rows[:2], measured


def candidate_command(base, budget, share):
    return [*base, "--moe-cache-mib", str(budget), "--moe-cache-cpu-share", share]


def candidate_schedule(base, ceiling, manual_budget, manual_share):
    baseline = (0, "0", "baseline-cache-off")
    probes = []
    for budget, label in ((max(1, ceiling // 2), "half"), (ceiling, "full")):
        for share in SHARES:
            tag = f"{label}-share-{share.replace('.', '_')}"
            probes.append((budget, share, tag))
    manual = ((manual_budget, manual_share, "manual-reference")
              if manual_budget is not None else None)
    return {
        "baseline": (*baseline, candidate_command(base, *baseline[:2])),
        "probes": [(budget, share, tag, candidate_command(base, budget, share))
                   for budget, share, tag in probes],
        "manual": ((*manual, candidate_command(base, *manual[:2])) if manual else None),
    }


def invoke(command, budget, tag, outdir):
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        error_type = error_message = None
    except OSError as exc:
        stdout = b""
        stderr = str(exc).encode(errors="replace")
        exit_code = None
        error_type = type(exc).__name__
        error_message = str(exc)
    (outdir / f"{tag}.stdout.bin").write_bytes(stdout)
    (outdir / f"{tag}.stderr.bin").write_bytes(stderr)
    stderr_text = stderr.decode(errors="replace")
    result = {"tag": tag, "command": command, "exit": exit_code,
              "eligible": False}
    policy = CACHE_POLICY_RE.search(stderr_text)
    if policy:
        result["active_cache_policy"] = policy.group(1).lower()
    if error_type:
        result.update(error_type=error_type, error_message=error_message,
                      stderr_diagnostic=stderr_text)
        result["reason"] = f"{error_type}: {error_message}"
        return result
    if exit_code:
        result["reason"] = f"native exit {exit_code}"
        return result
    try:
        scores, warm, measured = parse_rows(stderr)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    summary = CACHE_RE.search(stderr_text)
    allocated = float(summary.group(2)) if summary else None
    if budget and (allocated is None or allocated <= 0):
        result.update(reason="native cache allocation summary missing or zero",
                      scores=scores, warmup_rows=warm, measured_rows=measured,
                      allocated_mib=allocated)
        return result
    result.update(eligible=True, scores=scores, warmup_rows=warm,
                  measured_rows=measured, allocated_mib=allocated)
    return result


def aligned(result, checkpoint_ids):
    return result.get("eligible") and [row["n_kv"] for row in result["measured_rows"]] == checkpoint_ids


def report_failure(result, outdir, label):
    reason = result.get("reason", "native output was unusable")
    write_line(f"{label}: FAILED — {reason}", stream=sys.stderr, flush=True)
    if not result.get("error_type") and result.get("exit") in (0, None):
        write_line(f"Raw logs: {outdir / (result['tag'] + '.*.bin')}", stream=sys.stderr, flush=True)
        return
    for stream in ("stdout", "stderr"):
        data = (outdir / f"{result['tag']}.{stream}.bin").read_bytes()
        if data:
            write_line(f"--- native {stream} ({result['tag']}) ---", stream=sys.stderr, flush=True)
            sys.stderr.buffer.write(data)
            if not data.endswith(b"\n"):
                sys.stderr.buffer.write(b"\n")
            sys.stderr.buffer.flush()


def write_report(outdir, report, enabled):
    if enabled:
        (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def print_summary(baseline, probes, confirmation, manual, qualified):
    write_line("\nMeasured sweep comparison")
    write_line("Role                 Cache MiB  Share      PP tok/s      TG tok/s   TG Δ vs off  Status")
    write_line("-------------------- ---------  -----  ------------  ------------  -----------  ------")

    def row(role, budget, share, result, state):
        if result and result.get("eligible"):
            pp = f"{result['scores']['speed_pp']:.3f}"
            tg = f"{result['scores']['speed_tg']:.3f}"
            if baseline and baseline.get("eligible"):
                delta = 100 * (result["scores"]["speed_tg"] / baseline["scores"]["speed_tg"] - 1)
                delta_text = f"{delta:+.2f}%"
            else:
                delta_text = "—"
        else:
            pp = tg = delta_text = "—"
            if result and not result.get("eligible"):
                state = f"FAILED: {result.get('reason', 'unusable output')}"
        write_line(f"{role:<20} {budget:>9}  {share:>5}  {pp:>12}  {tg:>12}  {delta_text:>11}  {state}")

    row("Baseline", 0, "0", baseline, "baseline")
    for index, result in enumerate(probes, 1):
        row(f"Probe {index}/8", result["budget_mib"], result["cpu_share"], result, "measured")
    if confirmation is not None:
        row("Confirmation", confirmation["budget_mib"], confirmation["cpu_share"],
            confirmation, "confirmed" if qualified else "measured, below 5%")
    if manual is not None:
        row("Manual reference", manual["budget_mib"], manual["cpu_share"],
            manual, "outside ranking")


def extract_owned(argv):
    aliases = {
        "--moe-cache-mib": "budget", "--moe_cache_mib": "budget",
        "--moe-cache-cpu-share": "share", "--moe_cache_cpu_share": "share",
        "--output-format": "format", "--output_format": "format",
    }
    clean, manual_budget, manual_share = [], None, "0"
    i = 0
    while i < len(argv):
        token = argv[i]
        name, equals, inline = token.partition("=")
        kind = aliases.get(name)
        if not kind:
            clean.append(token)
            i += 1
            continue
        if equals:
            value = inline
        else:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"{name} needs a value")
            value = argv[i + 1]
            i += 1
        if not value:
            raise ValueError(f"{name} needs a value")
        if kind == "budget":
            try:
                parsed = int(value, 10)
            except ValueError:
                raise ValueError(f"{name} must be a nonnegative integer") from None
            if parsed < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            manual_budget = parsed if parsed > 0 else None
        elif kind == "share":
            manual_share = value
        i += 1
    return clean, manual_budget, manual_share


def runner_name(argv):
    """Find the first named llama runner before native model/alias arguments."""
    for token in argv:
        if token in {"--model", "--alias", "-a"}:
            break
        name = token.replace("\\", "/").rsplit("/", 1)[-1]
        if token.startswith("-") or "=" in token:
            continue
        name = name[:-4] if name.lower().endswith(".exe") else name
        if name.startswith("llama-"):
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Tune MoE cache settings with native llama-sweep-bench trials.")
    parser.add_argument("--budget-ceiling-mib", type=int, required=True, help="positive maximum cache budget in MiB")
    parser.add_argument("--out", type=Path, default=Path("sweep-tune-results"), help="directory for raw native output and report")
    parser.add_argument("--results-dir", dest="out", type=Path, help="alias for --out")
    parser.add_argument("--report-json", action="store_true",
                        help="write the full report to report.json")
    parser.add_argument("runner", nargs=argparse.REMAINDER, help="llama-sweep-bench argv after --")
    args = parser.parse_args()
    if args.budget_ceiling_mib < 2:
        parser.error("--budget-ceiling-mib must be at least 2 MiB")
    base = args.runner[1:] if args.runner[:1] == ["--"] else args.runner
    if not base:
        parser.error("supply llama-sweep-bench command after --")
    try:
        clean, manual_budget, manual_share = extract_owned(base)
    except ValueError as exc:
        parser.error(str(exc))
    named_runner = runner_name(clean)
    if named_runner != "llama-sweep-bench":
        if named_runner:
            parser.error(f"runner must name llama-sweep-bench, got {named_runner}")
        parser.error("runner must name the llama-sweep-bench executable (directly or with env/numactl)")
    base = [*clean, "--output-format", "jsonl"]
    schedule = candidate_schedule(base, args.budget_ceiling_mib, manual_budget, manual_share)
    args.out.mkdir(parents=True, exist_ok=True)
    ceiling = args.budget_ceiling_mib
    write_line("Sweep-only MoE-cache tuning", flush=True)
    write_line(f"Ceiling: {ceiling} MiB, probe shares: {', '.join(SHARES)}, score starts after two conditioning checkpoints.", flush=True)
    write_line(f"Results directory: {args.out}", flush=True)
    budget, share, tag, command = schedule["baseline"]
    write_line("Running baseline: cache-off, share 0", flush=True)
    baseline = invoke(command, budget, tag, args.out)
    if not baseline["eligible"]:
        if baseline.get("error_type"):
            status = "native launch failed"
        elif baseline["exit"]:
            status = "native run failed"
        else:
            status = "native validation failed"
        report = {"status": status, "baseline": baseline, "budget_ceiling_mib": ceiling}
        write_report(args.out, report, args.report_json)
        report_failure(baseline, args.out, "Baseline")
        write_line("No recommendation")
        write_line(f"Results directory: {args.out}")
        return 2
    baseline_ids = [row["n_kv"] for row in baseline["measured_rows"]]
    probes = []
    for index, (budget, share, tag, command) in enumerate(schedule["probes"], 1):
        write_line(f"Running probe {index}/8: {budget} MiB, share {share}", flush=True)
        result = invoke(command, budget, tag, args.out)
        result.update(budget_mib=budget, cpu_share=share)
        if result["eligible"] and not aligned(result, baseline_ids):
            result.update(eligible=False, reason="measured checkpoints do not match cache-off")
        if not result["eligible"]:
            report_failure(result, args.out, f"Probe {index}/8")
        probes.append(result)
    candidates = [r for r in probes if r["eligible"]]
    best = max(candidates, key=lambda r: r["scores"]["speed_tg"]) if candidates else None
    discovery_gain = (100 * (best["scores"]["speed_tg"] / baseline["scores"]["speed_tg"] - 1)
                      if best else None)
    confirmation = None
    if best and discovery_gain >= MIN_GAIN_PERCENT:
        write_line(f"Running confirmation: repeat {best['tag']} command ({best['budget_mib']} MiB, share {best['cpu_share']})", flush=True)
        confirmation = invoke(best["command"], best["budget_mib"],
                              "confirmation", args.out)
        confirmation.update(budget_mib=best["budget_mib"], cpu_share=best["cpu_share"])
        if confirmation["eligible"] and not aligned(confirmation, baseline_ids):
            confirmation.update(eligible=False, reason="confirmation checkpoints do not match cache-off")
        if not confirmation["eligible"]:
            report_failure(confirmation, args.out, "Confirmation")
    else:
        write_line(f"Confirmation skipped: best probe did not reach {MIN_GAIN_PERCENT:.0f}% TG discovery gain", flush=True)
    manual = None
    if manual_budget is not None:
        mbudget, mshare, mtag, mcommand = schedule["manual"]
        write_line(f"Running manual reference: {mbudget} MiB, share {mshare} (outside ranking)", flush=True)
        manual = invoke(mcommand, mbudget, mtag, args.out)
        manual.update(budget_mib=manual_budget, cpu_share=manual_share, reference_only=True)
        if manual["eligible"] and not aligned(manual, baseline_ids):
            manual.update(eligible=False, reason="manual checkpoints do not match cache-off")
        if not manual["eligible"]:
            report_failure(manual, args.out, "Manual reference")
        if manual["eligible"]:
            manual["tg_gain_vs_cache_off_percent"] = 100 * (manual["scores"]["speed_tg"] / baseline["scores"]["speed_tg"] - 1)
    else:
        write_line("Manual reference not requested", flush=True)
    confirmed_gain = None
    if confirmation and confirmation["eligible"]:
        confirmed_gain = 100 * (confirmation["scores"]["speed_tg"] / baseline["scores"]["speed_tg"] - 1)
    qualified = confirmation and confirmation["eligible"] and confirmed_gain >= MIN_GAIN_PERCENT
    if qualified:
        recommendation = {"setting": "cache", "budget_mib": best["budget_mib"], "cpu_share": best["cpu_share"]}
        status = "qualified cache recommendation"
    else:
        recommendation = {"setting": "cache-off"}
        if not candidates:
            status = "no eligible cached probes, recommend cache-off"
        elif discovery_gain <= 0 or (confirmed_gain is not None and confirmed_gain <= 0):
            status = "no measured TG benefit, recommend cache-off"
        else:
            status = "no cache candidate met the 5% confirmation rule, recommend cache-off"
    report = {"status": status, "runner": base, "budget_ceiling_mib": ceiling,
              "minimum_gain_percent": MIN_GAIN_PERCENT, "baseline": baseline,
              "probes": probes, "best_tested_cache_setting": best,
              "discovery_gain_percent": discovery_gain, "confirmation": confirmation,
              "confirmed_gain_percent": confirmed_gain, "recommendation": recommendation,
              "manual_reference": manual}
    write_report(args.out, report, args.report_json)
    print_summary(baseline, probes, confirmation, manual, qualified)
    write_line(f"Status: {status}")
    if qualified:
        policy = confirmation.get("active_cache_policy")
        detail = f", policy {policy}" if policy else ""
        write_line(f"Recommendation: cache {best['budget_mib']} MiB, share {best['cpu_share']}{detail}")
    else:
        write_line("Recommendation: cache-off")
        write_line(f"Cache-off PP/TG: {baseline['scores']['speed_pp']:.3f}/{baseline['scores']['speed_tg']:.3f} tok/s.")
    if qualified:
        write_line(f"Copyable winner command: {shlex.join(confirmation['command'])}")
    if confirmation and confirmation["eligible"] and manual and manual["eligible"]:
        relative = 100 * (confirmation["scores"]["speed_tg"] / manual["scores"]["speed_tg"] - 1)
        write_line(f"Confirmed vs manual TG: {confirmation['scores']['speed_tg']:.3f} vs {manual['scores']['speed_tg']:.3f} tok/s ({relative:+.2f}%).")
    write_line(f"Results directory: {args.out}")
    return 0 if qualified or candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
