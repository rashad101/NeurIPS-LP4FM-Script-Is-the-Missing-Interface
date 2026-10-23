"""``scriptgap`` command line interface.

Stages are separate subcommands so a run can be assembled piecewise, and every
stage takes ``--run-id`` so it can be pointed at an existing run:

    scriptgap download            # weights + datasets into the repo
    scriptgap calibrate           # fit the noise operators to attested data
    scriptgap build-data          # the five-condition suite
    scriptgap run-inference       # the GPU day
    scriptgap decompose           # H5
    scriptgap fertility           # H6 (CPU)
    scriptgap metrics             # aggregate -> parquet + CSVs
    scriptgap figures             # the six paper figures
    scriptgap all                 # everything, in order
    scriptgap export              # self-contained, checksummed archive of a run

``--run-id latest`` resumes the most recent run; ``--run-id <name>`` targets a
named one. Omitting it starts a fresh timestamped run.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Sequence

from . import ROOT, __version__
from .config import ALL_CONDITIONS, load_config
from .data.schema import TASKS
from .logging_utils import setup_logging, write_manifest
from .paths import RUNS_DIR, resolve_run


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="extra YAML config layered over configs/default.yaml (repeatable)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="dotted-key override, e.g. --set data.milu.n_per_language=10 (repeatable)",
    )
    parser.add_argument("--run-id", default=None, help="reuse a run id, or 'latest'")
    parser.add_argument("--run-prefix", default=None, help="label prefix for a new run id")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", action="append", default=[], help="restrict to model name(s)")
    parser.add_argument(
        "--task", action="append", default=[], choices=list(TASKS), help="restrict to task(s)"
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        choices=list(ALL_CONDITIONS),
        help="restrict to orthographic condition(s)",
    )
    parser.add_argument("--language", action="append", default=[], help="restrict to language(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriptgap",
        description="Script Is the Missing Interface — five-condition script-stratified evaluation",
    )
    parser.add_argument("--version", action="version", version=f"scriptgap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="prefetch model weights and datasets into the repo")
    _add_common(p)
    p.add_argument("--model", action="append", default=[], help="restrict to model name(s)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--skip-models", action="store_true", help="datasets only")
    p.add_argument("--skip-data", action="store_true", help="weights only")
    p.add_argument(
        "--dakshina",
        action="store_true",
        help="also fetch the Dakshina lexicons (attested human romanisation, ~2 GB read)",
    )

    p = sub.add_parser(
        "calibrate",
        help="fit condition-3 noise probabilities to attested human romanisation (Dakshina)",
    )
    _add_common(p)

    p = sub.add_parser("build-data", help="build base items and render five conditions")
    _add_common(p)
    p.add_argument("--task", action="append", default=[], choices=list(TASKS))
    p.add_argument("--force", action="store_true", help="rebuild base items and re-render")

    p = sub.add_parser("run-inference", help="generate and score predictions")
    _add_common(p)
    _add_filters(p)
    p.add_argument("--limit", type=int, default=None, help="cap items per shard (smoke tests)")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop cleanly after this many seconds; resume with the same --run-id",
    )
    p.add_argument("--include-disabled", action="store_true", help="also run models marked disabled")

    p = sub.add_parser("decompose", help="H5 failure-mode decomposition (LLM judge)")
    _add_common(p)
    p.add_argument("--model", action="append", default=[], help="restrict to model name(s)")

    p = sub.add_parser("fertility", help="H6 native-vs-romanised tokenisation (CPU only)")
    _add_common(p)
    p.add_argument("--task", action="append", default=[], choices=list(TASKS))
    p.add_argument("--include-disabled", action="store_true")

    p = sub.add_parser("metrics", help="aggregate predictions into parquet + CSVs")
    _add_common(p)
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="do not exclude items that failed the back-transliteration gate",
    )

    p = sub.add_parser("figures", help="render the six paper figures")
    _add_common(p)
    p.add_argument("--theme", default="light", choices=["light", "dark"])

    p = sub.add_parser("all", help="build-data -> inference -> decompose -> fertility -> metrics -> figures")
    _add_common(p)
    _add_filters(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--skip-decompose", action="store_true", help="skip the H5 judge (needs an API key)")
    p.add_argument("--no-export", action="store_true", help="skip the final archive step")
    p.add_argument("--include-disabled", action="store_true")

    p = sub.add_parser(
        "export", help="bundle a run into a self-contained, checksummed archive"
    )
    _add_common(p)
    p.add_argument("--out", default=None, help="archive path (default exports/scriptgap_<run>.tar.gz)")
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="also include raw safety completions — do not distribute the result",
    )
    p.add_argument("--no-compress", action="store_true", help="leave an uncompressed directory")

    p = sub.add_parser("verify-export", help="re-check an extracted export against its checksums")
    p.add_argument("path", help="extracted export directory (containing EXPORT.json)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub.add_parser("list-runs", help="list run directories, newest last")

    return parser


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------


def _models_for(cfg, args) -> list[Any]:
    from .models.registry import load_models, select_models

    specs = load_models(cfg.get("models"))
    return select_models(
        specs,
        only=list(args.model) if getattr(args, "model", None) else None,
        include_disabled=bool(getattr(args, "include_disabled", False)),
    )


def cmd_download(cfg, args, run) -> int:
    log = setup_logging("download", run, args.log_level)
    write_manifest(run, "download", {"config": cfg.to_dict(), "args": vars(args)})

    if not args.skip_models:
        from .models.registry import download_model, load_models, select_models

        specs = select_models(
            load_models(cfg.get("models")),
            only=list(args.model) or None,
            include_disabled=True,
        )
        for spec in specs:
            if not spec.enabled and not args.model:
                log.info("%s is disabled — skipping (pass --model %s to force)", spec.name, spec.name)
                continue
            download_model(spec, force=args.force)

    if not args.skip_data:
        from .data.build import build_base_items

        log.info("prefetching datasets by building base items")
        build_base_items(cfg, force=args.force)

    if args.dakshina or cfg.get("data.dakshina.enabled", False):
        from .data import dakshina

        root = dakshina.dakshina_root(cfg)
        if dakshina.is_available(root) and not args.force:
            log.info("Dakshina lexicons already present at %s", root)
        else:
            dakshina.download(root, str(cfg.get("data.dakshina.url", dakshina.DAKSHINA_URL)))

    log.info("download stage complete; everything lives under %s", ROOT)
    return 0


def cmd_calibrate(cfg, args, run) -> int:
    log = setup_logging("calibrate", run, args.log_level)
    write_manifest(run, "calibrate", {"config": cfg.get("data.dakshina")})

    from .analysis import calibrate

    payload = calibrate.run(cfg, run)
    log.info(
        "calibration complete over %d language(s); layer "
        "data/calibration/noisy_roman.yaml and rebuild with --force",
        len(payload["lexicons"]),
    )
    return 0


def cmd_build_data(cfg, args, run) -> int:
    log = setup_logging("build-data", run, args.log_level)
    write_manifest(run, "build-data", {"config": cfg.to_dict(), "args": vars(args)})

    from .data.build import build

    manifest = build(cfg, force=args.force, only=list(args.task) or None)
    log.info("build-data complete: %s cells", manifest["stats"]["cells"])
    return 0


def cmd_run_inference(cfg, args, run) -> int:
    log = setup_logging("inference", run, args.log_level)

    from .eval import isolate, run_inference
    from .models.registry import describe

    specs = _models_for(cfg, args)
    write_manifest(
        run,
        "inference",
        {"config": cfg.to_dict(), "args": vars(args), "models": describe(specs)},
    )
    log.info("run %s | models: %s", run.run_id, [s.name for s in specs])

    # One model in memory at a time. In-process teardown handles this for the
    # hf/anthropic backends; for vLLM a child process per model is the only way
    # to be certain the card is fully reclaimed between models.
    mode = str(cfg.get("inference.isolate_models", "auto"))
    default_backend = str(cfg.get("inference.backend", "vllm"))
    if isolate.should_isolate(mode, specs, default_backend):
        log.info(
            "isolate_models=%s: running %d models in separate processes so each "
            "one's GPU memory is reclaimed by the OS on exit",
            mode,
            len(specs),
        )
        return isolate.run_isolated(
            specs,
            base_argv=sys.argv[1:],
            run_id=run.run_id,
            max_seconds=args.max_seconds,
            continue_on_error=bool(cfg.get("inference.continue_on_model_error", True)),
        )

    summary = run_inference.run(
        cfg,
        run,
        specs,
        tasks=list(args.task) or None,
        conditions=list(args.condition) or None,
        languages=list(args.language) or None,
        limit_per_shard=args.limit,
        max_seconds=args.max_seconds,
    )
    if summary.get("budget_exhausted"):
        log.warning("stopped on the time budget; re-run with --run-id %s to continue", run.run_id)
        return 2
    return 0


def cmd_decompose(cfg, args, run) -> int:
    log = setup_logging("decompose", run, args.log_level)
    write_manifest(run, "decompose", {"config": cfg.get("decompose"), "args": vars(args)})

    from .eval import decompose

    summary = decompose.run(cfg, run, models=list(args.model) or None)
    log.info("decompose summary: %s", summary)
    return 0


def cmd_fertility(cfg, args, run) -> int:
    log = setup_logging("fertility", run, args.log_level)

    from .eval import fertility

    specs = _models_for(cfg, args)
    write_manifest(run, "fertility", {"config": cfg.get("fertility"), "args": vars(args)})
    summary = fertility.run(cfg, run, specs, tasks=list(args.task) or None)
    log.info("fertility summary: %s", summary)
    return 0


def cmd_metrics(cfg, args, run) -> int:
    log = setup_logging("metrics", run, args.log_level)

    from .analysis import metrics
    from .models.registry import load_models, specialised_models

    try:
        specialised = [s.name for s in specialised_models(load_models(cfg.get("models")))]
    except SystemExit:
        specialised = []
        log.warning("no model roster in the config — H2's variance statistic will be empty")

    write_manifest(run, "metrics", {"args": vars(args), "specialised_models": specialised})
    metrics.compute(cfg, run, specialised_models=specialised, gate=not args.no_gate)
    return 0


def cmd_figures(cfg, args, run) -> int:
    log = setup_logging("figures", run, args.log_level)

    from .analysis import figures

    formats = list(cfg.get("figures.formats", ["pdf", "png"]))
    dpi = int(cfg.get("figures.dpi", 200))
    write_manifest(run, "figures", {"formats": formats, "dpi": dpi, "theme": args.theme})

    result = figures.render_all(run, formats=formats, dpi=dpi, theme_name=args.theme)
    log.info("figures written to %s", run.figures)
    return 0 if result["figures"] else 1


def cmd_export(cfg, args, run) -> int:
    log = setup_logging("export", run, args.log_level)

    from .analysis import export

    archive = export.run(
        cfg,
        run,
        out=args.out,
        include_raw=args.include_raw,
        compress=not args.no_compress,
    )
    log.info("run %s archived -> %s", run.run_id, archive)
    return 0


def cmd_verify_export(_cfg, args, _run) -> int:
    from pathlib import Path

    from .analysis import export

    setup_logging("verify-export", None, args.log_level)
    result = export.verify(Path(args.path))
    return 0 if result["ok"] else 1


def cmd_all(cfg, args, run) -> int:
    log = setup_logging("all", run, args.log_level)
    log.info("full pipeline in run %s", run.run_id)

    stages: list[tuple[str, Any]] = [
        ("build-data", lambda: cmd_build_data(cfg, _ns(args, force=False, task=args.task), run)),
        ("inference", lambda: cmd_run_inference(cfg, args, run)),
    ]
    if not args.skip_decompose:
        stages.append(("decompose", lambda: cmd_decompose(cfg, _ns(args, model=args.model), run)))
    stages += [
        ("fertility", lambda: cmd_fertility(cfg, _ns(args, task=args.task), run)),
        ("metrics", lambda: cmd_metrics(cfg, _ns(args, no_gate=False), run)),
        ("figures", lambda: cmd_figures(cfg, _ns(args, theme="light"), run)),
        # Archive last: runs/ is gitignored, so this is what makes results
        # survive losing the machine they were produced on.
        ("export", lambda: cmd_export(cfg, _ns(args), run)),
    ]

    if getattr(args, "no_export", False):
        stages = [s for s in stages if s[0] != "export"]

    for name, fn in stages:
        log.info("---- stage: %s ----", name)
        code = fn()
        if code == 2 and name == "inference":
            log.warning("inference hit the time budget — stopping the pipeline here")
            return 2
        if code not in (0, None):
            log.error("stage %s exited with %s — stopping", name, code)
            return int(code)
    log.info("pipeline complete: results in %s", run.root)
    return 0


def _ns(base: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    """A shallow copy of ``base`` with per-stage defaults filled in."""
    data = dict(vars(base))
    data.setdefault("force", False)
    data.setdefault("no_gate", False)
    data.setdefault("theme", "light")
    data.setdefault("skip_models", False)
    data.setdefault("skip_data", False)
    data.setdefault("dakshina", False)
    data.setdefault("include_raw", False)
    data.setdefault("no_compress", False)
    data.setdefault("out", None)
    data.update(overrides)
    return argparse.Namespace(**data)


def cmd_list_runs(_cfg, _args, _run) -> int:
    if not RUNS_DIR.exists():
        print("no runs yet")
        return 0
    for path in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        shards = len(list((path / "predictions").rglob("*.jsonl"))) if (path / "predictions").exists() else 0
        figures = len(list((path / "figures").glob("*"))) if (path / "figures").exists() else 0
        print(f"{path.name:28s} shards={shards:<5d} figures={figures}")
    return 0


COMMANDS = {
    "download": cmd_download,
    "calibrate": cmd_calibrate,
    "build-data": cmd_build_data,
    "run-inference": cmd_run_inference,
    "decompose": cmd_decompose,
    "fertility": cmd_fertility,
    "metrics": cmd_metrics,
    "figures": cmd_figures,
    "all": cmd_all,
    "export": cmd_export,
    "verify-export": cmd_verify_export,
    "list-runs": cmd_list_runs,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-runs":
        return cmd_list_runs(None, args, None)
    if args.command == "verify-export":
        return cmd_verify_export(None, args, None)

    if args.no_progress:
        os.environ["SCRIPTGAP_NO_PROGRESS"] = "1"

    cfg = load_config(args.config, args.overrides)

    # Stages that only read an existing run default to resuming the latest one,
    # so `scriptgap metrics` after an inference run needs no flags.
    run_id = args.run_id
    if run_id is None and args.command in {
        "metrics",
        "figures",
        "decompose",
        "fertility",
        "export",
    }:
        from .paths import latest_run

        found = latest_run()
        if found is not None:
            run_id = found.run_id

    run = resolve_run(run_id, prefix=args.run_prefix)

    handler = COMMANDS[args.command]
    try:
        return int(handler(cfg, args, run) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted — progress is checkpointed; resume with "
              f"--run-id {run.run_id}", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
