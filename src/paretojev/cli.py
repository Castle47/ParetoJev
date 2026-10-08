from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from paretojev.config import load_yaml
from paretojev.data.kuairand import (
    KuaiRandSlateConfig,
    build_jev_input_rows,
    build_session_slates,
    export_jev_input_rows,
    export_slate_manifest,
    load_content_metadata,
    read_kuairand_log,
)
from paretojev.data.microlens import (
    MicroLensProtocol,
    cross_validated_hybrid,
    download_microlens,
    evaluate_score_column,
    load_prepared_candidates,
    prepare_microlens_candidates,
    save_microlens_protocol,
)
from paretojev.experiments import (
    add_reciprocal_rank_fusion,
    bootstrap_metric_intervals,
    complementarity_report,
    cross_validated_choice_temperature,
    cross_validated_logistic_stacking,
    cross_validated_platt_calibration,
    merge_decision_scores,
    pairwise_metric_differences,
    probability_diagnostics,
)
from paretojev.jev import (
    DecisionCache,
    OpenJevClient,
    ScoreJsonlConfig,
    TypeSafeJevClient,
    score_open_choice,
    score_open_noul,
    score_typesafe_choice,
    score_typesafe_noul,
    run_score_jsonl,
    select_objectives,
)
from paretojev.training import (
    PrecomputedTrainingConfig,
    SyntheticSmokeConfig,
    check_target_files,
    run_precomputed_training,
    run_synthetic_smoke,
)
from paretojev.utils.io import atomic_write_json, read_jsonl
from paretojev.utils.provenance import runtime_provenance


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _command_synthetic_smoke(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config) if args.config else {}
    smoke_raw = raw.get("smoke", raw)
    config = SyntheticSmokeConfig(**smoke_raw)
    result = run_synthetic_smoke(
        config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    _print_json(result)
    return 0 if result["status"] == "passed" else 2


def _command_train_precomputed(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config) if args.config else {}
    training_raw = raw.get("training", raw)
    config = PrecomputedTrainingConfig(**training_raw)
    directions = [int(value) for value in args.directions.split(",")]
    result = run_precomputed_training(
        args.train,
        args.validation,
        directions=directions,
        config=config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    _print_json(result)
    return 0


def _command_download_microlens(args: argparse.Namespace) -> int:
    files = download_microlens(
        args.directory,
        include_covers=args.covers,
        extract_covers=args.extract_covers,
        force=args.force,
    )
    _print_json({"downloaded": [str(path) for path in files]})
    return 0


def _command_prepare_microlens(args: argparse.Namespace) -> int:
    protocol = MicroLensProtocol(
        n_users=args.users,
        n_negatives=args.negatives,
        history_length=args.history,
        time_window_days=args.window_days,
        seed=args.seed,
        negative_sampling=args.negative_sampling,
    )
    frame = prepare_microlens_candidates(args.directory, protocol)
    save_microlens_protocol(frame, args.output_directory)
    metrics = {
        column: evaluate_score_column(
            frame, column, tie_policy=args.tie_policy, tie_seed=args.tie_seed
        )
        for column in ("pop", "title_sim", "itemcf")
    }
    report = {
        "protocol": asdict(protocol),
        "rows": len(frame),
        "users": int(frame["user"].nunique()),
        "baseline_metrics": metrics,
        "tie_policy": args.tie_policy,
        "tie_seed": args.tie_seed,
    }
    atomic_write_json(Path(args.output_directory) / "prepare_report.json", report)
    _print_json(report)
    return 0


def _command_evaluate_microlens(args: argparse.Namespace) -> int:
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1")
    if args.reference and args.bootstrap_samples == 0:
        raise ValueError("--reference requires --bootstrap-samples > 0")

    frame = load_prepared_candidates(args.input)
    excluded = {
        "user",
        "item",
        "position",
        "y",
        "target_timestamp",
        "hist_items",
        "hist_titles",
        "title",
    }
    base_score_columns = args.scores or [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    missing = set(base_score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    generated_columns: list[str] = []
    method_reports: dict[str, Any] = {}
    choice_probability_columns: set[str] = set()
    pointwise_probability_columns: set[str] = set()

    for column in args.choice_temperature or []:
        output_column = f"{column}_temp_cv"
        frame, folds = cross_validated_choice_temperature(
            frame,
            column,
            folds=args.folds,
            temperatures=args.temperature_grid,
            seed=args.seed,
            output_column=output_column,
        )
        generated_columns.append(output_column)
        choice_probability_columns.update((column, output_column))
        method_reports.setdefault("choice_temperature", {})[column] = {
            "output_column": output_column,
            "folds": folds,
        }

    for column in args.platt:
        output_column = f"{column}_platt_cv"
        frame, folds = cross_validated_platt_calibration(
            frame,
            column,
            folds=args.folds,
            seed=args.seed,
            output_column=output_column,
        )
        generated_columns.append(output_column)
        pointwise_probability_columns.update((column, output_column))
        method_reports.setdefault("platt", {})[column] = {
            "output_column": output_column,
            "folds": folds,
        }

    if args.rrf:
        frame = add_reciprocal_rank_fusion(
            frame,
            args.rrf,
            rank_constant=args.rank_constant,
            output_column=args.rrf_output_column,
        )
        generated_columns.append(args.rrf_output_column)
        method_reports["rrf"] = {
            "columns": args.rrf,
            "rank_constant": args.rank_constant,
            "output_column": args.rrf_output_column,
        }

    if args.hybrid:
        frame, folds = cross_validated_hybrid(
            frame,
            args.hybrid,
            folds=args.folds,
            weight_step=args.weight_step,
            objective=args.objective,
            seed=args.seed,
            output_column=args.hybrid_output_column,
            tie_policy=args.tie_policy,
            tie_seed=args.tie_seed,
        )
        generated_columns.append(args.hybrid_output_column)
        method_reports["hybrid"] = {
            "columns": args.hybrid,
            "output_column": args.hybrid_output_column,
            "folds": folds,
        }

    if args.stack:
        frame, stack_folds = cross_validated_logistic_stacking(
            frame,
            args.stack,
            folds=args.folds,
            inner_folds=args.inner_folds,
            c_values=args.stack_c_values,
            objective=args.objective,
            seed=args.seed,
            output_column=args.stack_output_column,
            tie_policy=args.tie_policy,
            tie_seed=args.tie_seed,
        )
        generated_columns.append(args.stack_output_column)
        method_reports["stacking"] = {
            "columns": args.stack,
            "output_column": args.stack_output_column,
            "folds": stack_folds,
        }

    score_columns = list(dict.fromkeys([*base_score_columns, *generated_columns]))
    metrics = {
        column: evaluate_score_column(
            frame, column, tie_policy=args.tie_policy, tie_seed=args.tie_seed
        )
        for column in score_columns
    }
    slate_sizes = frame.groupby("user", sort=False).size()
    if slate_sizes.nunique() != 1:
        raise ValueError(
            f"Candidate slate sizes are inconsistent: {sorted(slate_sizes.unique())}"
        )
    report: dict[str, Any] = {
        "users": int(frame["user"].nunique()),
        "candidate_count": int(slate_sizes.iloc[0]),
        "metrics": metrics,
        # Namespace alias consumed directly by configs/targets_research.yaml.
        "microlens": metrics,
        "tie_policy": args.tie_policy,
        "tie_seed": args.tie_seed,
        "methods": method_reports,
    }

    diagnostic_columns = list(
        dict.fromkeys(
            [
                *(args.probability_diagnostics or []),
                *sorted(choice_probability_columns),
                *sorted(pointwise_probability_columns),
            ]
        )
    )
    if diagnostic_columns:
        missing_diagnostics = set(diagnostic_columns) - set(frame.columns)
        if missing_diagnostics:
            raise ValueError(
                f"Missing probability diagnostic columns: {sorted(missing_diagnostics)}"
            )
        report["calibration"] = {
            column: probability_diagnostics(
                frame,
                column,
                n_bins=args.calibration_bins,
                normalize_as_choice=column in choice_probability_columns,
            )
            for column in diagnostic_columns
        }

    if args.bootstrap_samples > 0:
        report["confidence_intervals"] = {
            column: bootstrap_metric_intervals(
                frame,
                column,
                n_bootstrap=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed,
                tie_policy=args.tie_policy,
                tie_seed=args.tie_seed,
            )
            for column in score_columns
        }

    if args.reference:
        if args.reference not in score_columns:
            raise ValueError(
                f"Reference score {args.reference!r} is not among evaluated columns {score_columns}"
            )
        report["pairwise_vs_reference"] = {
            column: pairwise_metric_differences(
                frame,
                column,
                args.reference,
                n_bootstrap=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed,
                tie_policy=args.tie_policy,
                tie_seed=args.tie_seed,
            )
            for column in score_columns
            if column != args.reference
        }

    complementarity_columns = args.complementarity
    if not complementarity_columns:
        complementarity_columns = args.hybrid or args.stack or args.rrf
    if complementarity_columns and len(complementarity_columns) >= 2:
        report["complementarity"] = complementarity_report(
            frame,
            complementarity_columns,
            tie_policy=args.tie_policy,
            tie_seed=args.tie_seed,
        )

    report["provenance"] = runtime_provenance(
        input_paths=[args.input],
        config={
            "scores": score_columns,
            "choice_temperature": args.choice_temperature,
            "temperature_grid": args.temperature_grid,
            "platt": args.platt,
            "probability_diagnostics": diagnostic_columns,
            "calibration_bins": args.calibration_bins,
            "hybrid": args.hybrid,
            "stack": args.stack,
            "rrf": args.rrf,
            "folds": args.folds,
            "inner_folds": args.inner_folds,
            "stack_c_values": args.stack_c_values,
            "weight_step": args.weight_step,
            "rank_constant": args.rank_constant,
            "objective": args.objective,
            "tie_policy": args.tie_policy,
            "tie_seed": args.tie_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": args.confidence,
            "reference": args.reference,
            "complementarity": complementarity_columns,
        },
    )

    if generated_columns or args.output_scores:
        output_scores = Path(
            args.output_scores or Path(args.input).with_name("evaluated_scores.csv")
        )
        serializable = frame.copy()
        if "hist_items" in serializable:
            serializable["hist_items"] = serializable["hist_items"].map(json.dumps)
        if "hist_titles" in serializable:
            serializable["hist_titles"] = serializable["hist_titles"].map(json.dumps)
        output_scores.parent.mkdir(parents=True, exist_ok=True)
        serializable.to_csv(output_scores, index=False)
        report["output_scores"] = str(output_scores)
    if args.output:
        atomic_write_json(args.output, report)
    _print_json(report)
    return 0


def _command_prepare_kuairand(args: argparse.Namespace) -> int:
    frame = read_kuairand_log(args.log, nrows=args.nrows)
    config = KuaiRandSlateConfig(
        slate_size=args.slate_size,
        min_slate_size=args.min_slate_size,
        session_gap_minutes=args.session_gap_minutes,
        stride=args.stride,
    )
    slates = build_session_slates(frame, config)
    slates.save(args.output)
    if args.manifest:
        export_slate_manifest(slates, args.manifest, max_rows=args.manifest_rows)
    if args.jev_input:
        if not args.content_directory:
            raise ValueError("--content-directory is required with --jev-input")
        content = load_content_metadata(args.content_directory)
        rows = build_jev_input_rows(
            slates, frame, content, history_length=args.history_length, max_rows=args.manifest_rows
        )
        export_jev_input_rows(rows, args.jev_input)
    report = {
        "config": asdict(config),
        "input_rows": len(frame),
        "slates": len(slates.user_ids),
        "unique_users": int(len(set(slates.user_ids.tolist()))),
        "mean_valid_candidates": float(slates.mask.sum(axis=1).mean()),
        "objective_names": list(slates.objective_names),
        "output": str(args.output),
        "manifest": str(args.manifest) if args.manifest else None,
        "jev_input": str(args.jev_input) if args.jev_input else None,
    }
    if args.report:
        atomic_write_json(args.report, report)
    _print_json(report)
    return 0


def _command_score_jsonl(args: argparse.Namespace) -> int:
    cache = DecisionCache(args.cache)
    objectives = select_objectives(args.objectives)
    if args.provider == "typesafe":
        client: Any = TypeSafeJevClient(
            api_key=args.api_key,
            base_url=args.base_url or "https://api.typesafe.ai",
            model=args.model or "jev-1.13.0",
            cache=cache,
            timeout_seconds=args.timeout_seconds,
        )
        score_function = score_typesafe_choice if args.mode == "choice" else score_typesafe_noul
    else:
        client = OpenJevClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model or "autotrust/JEV-27B",
            cache=cache,
            timeout_seconds=args.timeout_seconds,
        )
        score_function = score_open_choice if args.mode == "choice" else score_open_noul

    config = ScoreJsonlConfig(
        provider=args.provider,
        mode=args.mode,
        objective_names=tuple(objective.name for objective in objectives),
        noul_scope=args.noul_scope,
        resume=args.resume and not args.overwrite,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        durable_every=args.durable_every,
        start_index=args.start_index,
        max_new_rows=args.max_rows,
    )
    report = run_score_jsonl(
        args.input,
        args.output,
        client=client,
        score_function=score_function,
        objectives=objectives,
        config=config,
        error_path=args.errors,
        report_path=args.report,
        cache_count=cache.count,
    )
    _print_json(report)
    return 0 if report["status"] == "completed" else 2


def _command_merge_decision_scores(args: argparse.Namespace) -> int:
    frame = load_prepared_candidates(args.candidates)
    rows = read_jsonl(args.scores)
    merged = merge_decision_scores(
        frame,
        rows,
        objective=args.objective,
        slate_column=args.slate_column,
        item_column=args.item_column,
        score_column=args.score_column,
    )
    serializable = merged.copy()
    for column in ("hist_items", "hist_titles"):
        if column in serializable:
            serializable[column] = serializable[column].map(json.dumps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable.to_csv(output, index=False)
    _print_json({"rows": len(merged), "output": str(output), "score_column": args.score_column or f"jev_{args.objective}"})
    return 0


def _command_check_targets(args: argparse.Namespace) -> int:
    report = check_target_files(args.metrics, args.targets, args.output)
    _print_json(report)
    return 0 if report["passed"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paretojev",
        description="Decision-model-assisted multi-objective recommendation reranking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("synthetic-smoke", help="Run the end-to-end CPU smoke experiment")
    smoke.add_argument("--config")
    smoke.add_argument("--output", default="artifacts/smoke_metrics.json")
    smoke.add_argument("--checkpoint", default="artifacts/smoke_model.pt")
    smoke.set_defaults(func=_command_synthetic_smoke)

    train = subparsers.add_parser("train-precomputed", help="Train ParetoJev from feature-complete NPZ slates")
    train.add_argument("train")
    train.add_argument("validation")
    train.add_argument("--directions", required=True, help="Comma-separated +1/-1, e.g. 1,1,1,1,1,-1")
    train.add_argument("--config")
    train.add_argument("--output", default="artifacts/train_metrics.json")
    train.add_argument("--checkpoint", default="artifacts/paretojev.pt")
    train.set_defaults(func=_command_train_precomputed)

    download = subparsers.add_parser("download-microlens", help="Download official MicroLens-100K files")
    download.add_argument("directory")
    download.add_argument("--covers", action="store_true")
    download.add_argument("--extract-covers", action="store_true")
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=_command_download_microlens)

    prepare_micro = subparsers.add_parser(
        "prepare-microlens", help="Build the deterministic 200-user candidate protocol"
    )
    prepare_micro.add_argument("directory")
    prepare_micro.add_argument("output_directory")
    prepare_micro.add_argument("--users", type=int, default=200)
    prepare_micro.add_argument("--negatives", type=int, default=19)
    prepare_micro.add_argument("--history", type=int, default=5)
    prepare_micro.add_argument("--window-days", type=int, default=3)
    prepare_micro.add_argument("--seed", type=int, default=0)
    prepare_micro.add_argument(
        "--negative-sampling",
        choices=["upstream_occurrence", "uniform_unique"],
        default="upstream_occurrence",
        help="Use upstream occurrence-weighted sampling for published-demo comparability",
    )
    prepare_micro.add_argument(
        "--tie-policy",
        choices=["stable", "upstream_jitter", "random_jitter"],
        default="upstream_jitter",
    )
    prepare_micro.add_argument("--tie-seed", type=int, default=1)
    prepare_micro.set_defaults(func=_command_prepare_microlens)

    evaluate_micro = subparsers.add_parser(
        "evaluate-microlens", help="Evaluate score columns and an optional cross-validated hybrid"
    )
    evaluate_micro.add_argument("input")
    evaluate_micro.add_argument("--scores", nargs="+")
    evaluate_micro.add_argument(
        "--choice-temperature",
        nargs="+",
        help="Choice probability columns to calibrate with user-CV temperature scaling",
    )
    evaluate_micro.add_argument(
        "--temperature-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    evaluate_micro.add_argument(
        "--platt",
        nargs="+",
        default=[],
        help="Pointwise Noul/binary probability columns to calibrate with user-CV Platt scaling",
    )
    evaluate_micro.add_argument(
        "--probability-diagnostics",
        nargs="+",
        help="Additional probability columns for Brier/ECE/NLL reporting",
    )
    evaluate_micro.add_argument("--calibration-bins", type=int, default=15)
    evaluate_micro.add_argument("--hybrid", nargs="+")
    evaluate_micro.add_argument("--hybrid-output-column", default="hybrid_cv")
    evaluate_micro.add_argument("--stack", nargs="+")
    evaluate_micro.add_argument("--stack-output-column", default="stack_cv")
    evaluate_micro.add_argument(
        "--stack-c-values", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0]
    )
    evaluate_micro.add_argument("--inner-folds", type=int, default=4)
    evaluate_micro.add_argument("--rrf", nargs="+")
    evaluate_micro.add_argument("--rrf-output-column", default="rrf")
    evaluate_micro.add_argument("--rank-constant", type=float, default=60.0)
    evaluate_micro.add_argument("--folds", type=int, default=5)
    evaluate_micro.add_argument("--weight-step", type=float, default=0.05)
    evaluate_micro.add_argument("--objective", choices=["auc", "ndcg@10", "balanced"], default="balanced")
    evaluate_micro.add_argument("--reference")
    evaluate_micro.add_argument("--complementarity", nargs="+")
    evaluate_micro.add_argument("--bootstrap-samples", type=int, default=5000)
    evaluate_micro.add_argument("--confidence", type=float, default=0.95)
    evaluate_micro.add_argument("--seed", type=int, default=0)
    evaluate_micro.add_argument(
        "--tie-policy",
        choices=["stable", "upstream_jitter", "random_jitter"],
        default="upstream_jitter",
    )
    evaluate_micro.add_argument("--tie-seed", type=int, default=1)
    evaluate_micro.add_argument("--output")
    evaluate_micro.add_argument("--output-scores")
    evaluate_micro.set_defaults(func=_command_evaluate_microlens)

    prepare_kuai = subparsers.add_parser(
        "prepare-kuairand", help="Build fixed-width exposed-session pseudo-slates"
    )
    prepare_kuai.add_argument("log")
    prepare_kuai.add_argument("output")
    prepare_kuai.add_argument("--slate-size", type=int, default=20)
    prepare_kuai.add_argument("--min-slate-size", type=int, default=5)
    prepare_kuai.add_argument("--session-gap-minutes", type=int, default=30)
    prepare_kuai.add_argument("--stride", type=int)
    prepare_kuai.add_argument("--nrows", type=int)
    prepare_kuai.add_argument("--manifest")
    prepare_kuai.add_argument("--manifest-rows", type=int)
    prepare_kuai.add_argument("--jev-input")
    prepare_kuai.add_argument("--content-directory")
    prepare_kuai.add_argument("--history-length", type=int, default=10)
    prepare_kuai.add_argument("--report")
    prepare_kuai.set_defaults(func=_command_prepare_kuairand)

    score = subparsers.add_parser(
        "score-jsonl", help="Score JSONL recommendation slates with official or open Jev"
    )
    score.add_argument("input")
    score.add_argument("output")
    score.add_argument("--provider", choices=["typesafe", "open"], default="typesafe")
    score.add_argument("--mode", choices=["choice", "noul"], default="choice")
    score.add_argument("--base-url")
    score.add_argument("--api-key")
    score.add_argument("--model")
    score.add_argument(
        "--objectives",
        nargs="+",
        help="Subset such as: click long_view like (default: all objectives)",
    )
    score.add_argument(
        "--noul-scope",
        choices=["pointwise", "slate"],
        default="pointwise",
        help="Pointwise hides competing candidates from each Noul decision",
    )
    score.add_argument("--cache", default="artifacts/jev_cache.sqlite3")
    score.add_argument("--timeout-seconds", type=float, default=300.0)
    score.add_argument("--max-rows", type=int, help="Maximum number of new successful slates")
    score.add_argument("--start-index", type=int, default=0)
    score.add_argument("--durable-every", type=int, default=1)
    score.add_argument("--continue-on-error", action="store_true")
    score.add_argument("--errors")
    score.add_argument("--report")
    score.add_argument("--overwrite", action="store_true")
    score.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible existing JSONL output (default: true)",
    )
    score.set_defaults(func=_command_score_jsonl)

    merge = subparsers.add_parser("merge-decision-scores", help="Merge score-jsonl output into candidate rows")
    merge.add_argument("candidates")
    merge.add_argument("scores")
    merge.add_argument("output")
    merge.add_argument("--objective", default="click")
    merge.add_argument("--slate-column", default="user")
    merge.add_argument("--item-column", default="item")
    merge.add_argument("--score-column")
    merge.set_defaults(func=_command_merge_decision_scores)

    targets = subparsers.add_parser("check-targets", help="Evaluate a metrics JSON against target-mode gates")
    targets.add_argument("metrics")
    targets.add_argument("targets")
    targets.add_argument("--output")
    targets.set_defaults(func=_command_check_targets)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args))
    except Exception as exc:  # CLI boundary: concise message plus non-zero exit.
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            raise
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
