"""
core/cli.py
Headless batch analysis, without Qt.

    python -m core.cli run --plan plan.json --out DIR [--workers N]
                           [--paradigm NAME] [--videos PATH ...]
                           [--sample-sheet SHEET.csv] [--bin-seconds N]

``plan.json`` is a saved batch plan (File > Save Configuration, or the
``run_plan.json`` of an earlier run) or a single template. A plan names its
own videos; a template needs ``--videos`` (files and/or folders). The run
folder, batch summary, manifest and ``run_plan.json`` are the ones the GUI
writes, because both drive the same ``core.batch_runner``.

Exit status: 0 success, 1 some video failed, 2 bad arguments or plan,
130 cancelled.
"""
import argparse
import multiprocessing
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.batch_plan_keys import match_videos_to_plan
from core.batch_runner import BatchRunner, BatchRunOutcome, default_worker_count
from core.models import BatchSettings
from core.output_contract import COMMON_BATCH_HEADERS
from core.run_outputs import (
    apply_effective_parameters, create_run_folder, output_base_names,
    run_timestamp,
)
from core.time_bins import TIME_BIN_HEADERS
from core.sample_sheet import (
    SampleMetadata, SampleSheetError, load_sample_sheet,
    metadata_for_videos, reserved_column_conflicts,
)
from mazes import AVAILABLE_MAZES, create_maze
from mazes.base_maze import Maze
from mazes.tst import TST
from preflight_checker import PreflightChecker

VIDEO_EXTENSIONS = frozenset({'.mp4', '.avi', '.mov', '.mkv'})
# Result folders hold validation videos and checkpoints; never re-import them.
RUN_FOLDER_PREFIX = "analysis_run_"

EXIT_OK = 0
EXIT_FAILED_VIDEOS = 1
EXIT_BAD_INPUT = 2
EXIT_CANCELLED = 130

# Console progress is printed each time the batch advances by this many percent.
PROGRESS_PRINT_STEP = 5


class CliError(Exception):
    """A problem with the arguments or the plan; reported, exit status 2."""


# ----------------------------------------------------------------------
# Reporter
# ----------------------------------------------------------------------
class ConsoleReporter:
    """Batch reporter that prints to stdout."""

    def __init__(self, stream=None, quiet: bool = False):
        self._stream = stream or sys.stdout
        self._quiet = quiet
        self._last_pct = -PROGRESS_PRINT_STEP

    def _print(self, text: str) -> None:
        print(text, file=self._stream, flush=True)

    def progress(self, pct: int, main: str, sub: str = "") -> None:
        if self._quiet or pct < self._last_pct + PROGRESS_PRINT_STEP:
            return
        self._last_pct = pct
        self._print(f"[{pct:3d}%] {main} - {sub}" if sub else f"[{pct:3d}%] {main}")

    def log(self, message: str) -> None:
        if not self._quiet:
            self._print(message)

    def video_status(self, video_path: Path, status: str, message: str) -> None:
        if status in ("failed", "warning") or not self._quiet:
            suffix = f": {message}" if message else ""
            self._print(f"  {video_path.name} -> {status}{suffix}")


# ----------------------------------------------------------------------
# Plan loading
# ----------------------------------------------------------------------
def find_videos(paths: Sequence[Path]) -> List[Path]:
    """Expand files and folders (one level) into a sorted video list."""
    found: List[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found.extend(sorted(
                child for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
            ))
        elif path.is_file():
            found.append(path)
        else:
            raise CliError(f"Video path does not exist: {path}")
    unique: List[Path] = []
    for video in found:
        if video not in unique:
            unique.append(video)
    return unique


def _resolve_key(key: str, plan_file: Path) -> Path:
    candidate = Path(key)
    return candidate if candidate.is_absolute() else plan_file.parent / candidate


def load_analysis_plan(
    plan_file: Path, videos: Sequence[Path] = ()
) -> Dict[Path, BatchSettings]:
    """{video: settings} from a batch plan or a template."""
    kind = BatchSettings.detect_file_type(plan_file)
    if kind == "unknown":
        raise CliError(f"Cannot read a plan or template from {plan_file}")
    try:
        if kind == "batch_plan":
            by_key = BatchSettings.load_batch_plan(plan_file)
            if videos:
                match = match_videos_to_plan(videos, by_key, plan_file)
                if match.unmatched:
                    details = "; ".join(
                        f"{video.name} ({reason})"
                        for video, reason in match.unmatched
                    )
                    raise CliError(f"Videos not matched to the plan: {details}")
                return {
                    video: by_key[key].copy()
                    for video, key in match.matched.items()
                }
            plan = {
                _resolve_key(key, plan_file).resolve(): settings
                for key, settings in by_key.items()
            }
        else:
            if not videos:
                raise CliError(
                    f"{plan_file.name} is a single template; "
                    "pass the videos to analyse with --videos"
                )
            template = BatchSettings.load_from_file(plan_file)
            plan = {Path(video).resolve(): template.copy() for video in videos}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise CliError(f"Cannot load {plan_file}: {exc}") from exc
    missing = [video for video in plan if not video.is_file()]
    if missing:
        raise CliError(
            "Videos named by the plan were not found: "
            + ", ".join(str(video) for video in missing)
        )
    if not plan:
        raise CliError(f"{plan_file.name} contains no videos")
    return plan


def resolve_maze(
    plan: Dict[Path, BatchSettings], paradigm: Optional[str]
) -> Maze:
    """The configured maze for a plan, from its paradigm id or ``--paradigm``."""
    plan_paradigms = {s.paradigm_id for s in plan.values() if s.paradigm_id}
    if len(plan_paradigms) > 1:
        raise CliError(
            "The plan mixes paradigms: " + ", ".join(sorted(plan_paradigms))
        )
    plan_paradigm = next(iter(plan_paradigms), None)
    chosen = _match_paradigm(paradigm) if paradigm else None
    if chosen and plan_paradigm and chosen != plan_paradigm:
        raise CliError(
            f"--paradigm {paradigm!r} does not match the plan's paradigm "
            f"{plan_paradigm!r}"
        )
    name = chosen or plan_paradigm
    if name is None:
        raise CliError(
            "The plan does not name a paradigm; pass --paradigm "
            f"({', '.join(AVAILABLE_MAZES)})"
        )
    maze = create_maze(name)
    first = next(iter(plan.values()))
    try:
        if first.algorithm_parameters:
            maze.configure_algorithm_parameters(first.algorithm_parameters)
    except (TypeError, ValueError) as exc:
        raise CliError(f"Incompatible configuration: {exc}") from exc
    return maze


def _match_paradigm(text: str) -> str:
    wanted = text.strip().lower()
    names = list(AVAILABLE_MAZES)
    for name in names:
        if name.lower() == wanted:
            return name
    partial = [name for name in names if wanted in name.lower()]
    if len(partial) == 1:
        return partial[0]
    raise CliError(
        f"Unknown paradigm {text!r}; choose from: {', '.join(names)}"
    )


def load_sample_metadata(
    sheet_file: Path, plan: Dict[Path, BatchSettings], maze: Maze
) -> SampleMetadata:
    """Sample-sheet metadata for every video of the plan, or a CliError.

    Every video must match exactly one row: guessing which animal a video
    shows would silently corrupt the data, so a run with missing or
    ambiguous rows is refused.
    """
    try:
        sheet = load_sample_sheet(sheet_file)
    except SampleSheetError as exc:
        raise CliError(str(exc)) from exc
    conflicts = reserved_column_conflicts(
        sheet.columns,
        list(COMMON_BATCH_HEADERS) + list(maze.get_batch_summary_headers())
        + ["video_filename", "paradigm"] + list(TIME_BIN_HEADERS),
    )
    if conflicts:
        raise CliError(
            f"{sheet_file.name}: column(s) {', '.join(conflicts)} clash with "
            "output columns (batch summary or time bins); rename them in "
            "the sheet"
        )
    match = metadata_for_videos(sheet, sheet_file, list(plan))
    if not match.is_complete:
        raise CliError(
            f"Videos not matched to {sheet_file.name}: "
            + "; ".join(match.problems())
        )
    return match.metadata


def plan_issues(plan: Dict[Path, BatchSettings], maze: Maze) -> List[str]:
    """Plan-level problems the GUI also refuses to start with."""
    issues: List[str] = []
    required = [name for name, _ in maze.get_roi_definitions()]
    expected_parameters = None
    for video, settings in plan.items():
        if required:
            _, roi_issues = settings.roi_manager.validate_rois(required)
            issues.extend(f"{video.name}: {issue}" for issue in roi_issues)
        if isinstance(maze, TST):
            extra = set(settings.roi_manager.rois) - set(required)
            if extra:
                issues.append(
                    f"{video.name}: unexpected TST ROI categories: "
                    f"{', '.join(sorted(extra))}"
                )
            for category in required:
                count = len(settings.roi_manager.rois.get(category, []))
                if count != 1:
                    issues.append(
                        f"{video.name}: {category} must contain exactly one "
                        f"ROI, found {count}"
                    )
            unknown = set(settings.per_roi_times) - set(required)
            if unknown:
                issues.append(
                    f"{video.name}: time ranges reference unknown mice: "
                    f"{', '.join(sorted(unknown))}"
                )
        if settings.paradigm_id and settings.paradigm_id != maze.name:
            issues.append(
                f"{video.name}: configuration is for "
                f"'{settings.paradigm_id}', not '{maze.name}'"
            )
        if expected_parameters is None:
            expected_parameters = settings.algorithm_parameters
        elif settings.algorithm_parameters != expected_parameters:
            issues.append(
                f"{video.name}: paradigm parameters differ from the rest of "
                "the batch; mixed-parameter batches are not supported"
            )
    return issues


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def run_plan(
    plan: Dict[Path, BatchSettings],
    maze: Maze,
    output_dir: Path,
    workers: Optional[int],
    reporter: ConsoleReporter,
    cancel: threading.Event,
    sample_metadata: Optional[SampleMetadata] = None,
    bin_seconds: float = 0.0,
) -> Tuple[BatchRunOutcome, Path]:
    """Run an already validated plan; returns the outcome and the run folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_effective_parameters(plan, maze)
    run_dir = create_run_folder(output_dir)
    count = workers if workers else default_worker_count(len(plan))
    runner = BatchRunner(
        plan,
        output_dir,
        run_dir,
        maze,
        output_base_names(list(plan)),
        reporter,
        cancel,
        workers=count,
        started_at=run_timestamp(),
        sample_metadata=sample_metadata,
        bin_seconds=bin_seconds,
    )
    return runner.run(), run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m core.cli",
        description="Headless Behavioral Maze Analyzer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Analyse the videos of a saved batch plan or template."
    )
    run.add_argument("--plan", required=True, type=Path,
                     help="batch plan or template (.json)")
    run.add_argument("--out", required=True, type=Path,
                     help="output folder; the run folder is created inside")
    run.add_argument("--workers", type=int, default=None,
                     help="videos analysed at once (default: "
                          "min(cpu/4, videos, ram_gb/3), at least 1)")
    run.add_argument("--paradigm", default=None,
                     help="paradigm name, needed when the plan has none")
    run.add_argument("--videos", nargs="+", type=Path, default=[],
                     help="videos or folders: required for a template, "
                          "optional subset selector for a batch plan")
    run.add_argument("--sample-sheet", type=Path, default=None,
                     help="CSV of animal metadata: first column 'video', "
                          "other columns (animal_id, group, ...) are added "
                          "to the batch summary, summary.json and manifest")
    run.add_argument("--bin-seconds", type=float, default=0.0,
                     help="also write per-time-bin metrics (data/<video>_"
                          "time_bins.csv and batch_time_bins.csv) with bins "
                          "of this many seconds; 0 (default) turns it off")
    run.add_argument("--quiet", action="store_true",
                     help="print only failures, warnings and the result")
    return parser


def _command_run(args, reporter: ConsoleReporter) -> int:
    if args.workers is not None and args.workers < 1:
        raise CliError("--workers must be at least 1")
    if args.bin_seconds < 0:
        raise CliError("--bin-seconds cannot be negative (0 turns bins off)")
    videos = find_videos(args.videos) if args.videos else []
    plan = load_analysis_plan(args.plan, videos)
    maze = resolve_maze(plan, args.paradigm)
    issues = plan_issues(plan, maze)
    apply_effective_parameters(plan, maze)
    sample_metadata = (
        load_sample_metadata(args.sample_sheet, plan, maze)
        if args.sample_sheet else None
    )
    preflight_issues, warnings = PreflightChecker().validate_batch(
        plan, args.out, paradigm_name=maze.name
    )
    issues.extend(preflight_issues)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if issues:
        raise CliError(
            "Cannot start the analysis:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )

    cancel = threading.Event()

    def request_stop(_signum, _frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        print("Stopping after the current frames - saving progress "
              "(press Ctrl+C again to abort)...", file=sys.stderr)
        cancel.set()

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        outcome, run_dir = run_plan(
            plan, maze, args.out, args.workers, reporter, cancel,
            sample_metadata=sample_metadata,
            bin_seconds=args.bin_seconds,
        )
    finally:
        signal.signal(signal.SIGINT, previous)

    print(f"Run folder: {run_dir}")
    print(
        f"Processed {len(outcome.results)}/{len(plan)} video(s); "
        f"{len(outcome.failures)} failed, "
        f"{len(outcome.quality_warnings)} quality warning(s)."
    )
    for path, message in outcome.failures:
        print(f"  FAILED {path.name}: {message}", file=sys.stderr)
    if outcome.cancelled:
        return EXIT_CANCELLED
    return EXIT_FAILED_VIDEOS if outcome.failures else EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    reporter = ConsoleReporter(quiet=getattr(args, "quiet", False))
    try:
        if args.command == "run":
            return _command_run(args, reporter)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    return EXIT_BAD_INPUT


if __name__ == "__main__":
    # Needed for the 'spawn' child processes of a parallel batch (and for a
    # frozen build); harmless otherwise.
    multiprocessing.freeze_support()
    sys.exit(main())
