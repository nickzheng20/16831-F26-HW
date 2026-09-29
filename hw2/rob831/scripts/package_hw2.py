"""Validate all official HW2 runs and build the submission archive.

Run from the project root with ``python -m rob831.scripts.package_hw2``.
The archive preserves each complete run's directory name under ``run_logs/``.
An incomplete run with training scalars is an error, even if another attempt
for the same configuration completed. Empty startup failures are reported and
omitted. Use ``--check-only`` to validate the inputs without creating an archive.
"""

import argparse
import csv
import math
from pathlib import Path
import re
import sys
import tempfile
import zipfile


ITERATIONS = {1: 150, 2: 100, 3: 100, 4: 100, 5: 300}
MINIMUM_RUNS = {1: 6, 2: 1, 3: 1, 4: 13, 5: 4}
SIZE_LIMIT = 15_000_000
RESULT_SUFFIXES = {".pdf", ".png", ".csv", ".json", ".sh", ".md", ".py"}
EXCLUDED_PARTS = {"__pycache__", "startup_failure", "policy_demo"}
REQUIRED_FILES = (
    "README.md", "requirements.txt", "setup.py", ".latexmkrc",
    "hw2_submission.tex", "hw2_submission.pdf", "core_commands.sh",
)


def excluded(path):
    return bool(EXCLUDED_PARTS.intersection(path.parts)) or path.suffix == ".log"


def complete_runs(data_dir):
    # Import lazily so --help works even before the homework dependencies exist.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not data_dir.is_dir():
        raise ValueError(f"Log directory does not exist: {data_dir}")
    runs, errors = [], []
    counts = {question: 0 for question in ITERATIONS}
    for run in sorted(data_dir.iterdir()):
        match = re.match(r"^q([1-5])_", run.name)
        if not match or not run.is_dir():
            continue
        question = int(match.group(1))
        try:
            events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
            scalar_tags = events.Tags()["scalars"]
            has_scalars = any(events.Scalars(tag) for tag in scalar_tags)
            # File headers without summaries are produced by failed startups.
            # Other summaries still represent data and must not be lost silently.
            has_other_data = any(
                events.Tags().get(tag)
                for tag in ("tensors", "images", "audio", "histograms", "distributions")
            )
            if not has_scalars and not has_other_data:
                print(f"Skipping empty startup run: {run.name}")
                continue
            if "Eval_AverageReturn" not in scalar_tags:
                errors.append(f"{run.name}: logged data but no Eval_AverageReturn")
                continue
            points = events.Scalars("Eval_AverageReturn")
            expected = ITERATIONS[question]
            if [point.step for point in points] != list(range(expected)):
                errors.append(
                    f"{run.name}: expected steps 0..{expected - 1}; "
                    f"found {len(points)} evaluation points"
                )
                continue
            if not all(math.isfinite(point.value) for point in points):
                errors.append(f"{run.name}: non-finite evaluation return")
                continue
            runs.append(run)
            counts[question] += 1
        except Exception as error:
            errors.append(f"{run.name}: cannot validate logs: {error}")
    for question, minimum in MINIMUM_RUNS.items():
        if counts[question] < minimum:
            errors.append(
                f"Q{question}: found {counts[question]} complete runs; "
                f"expected at least {minimum}"
            )
    if errors:
        raise ValueError("Run validation failed:\n  " + "\n  ".join(errors))
    for question, count in counts.items():
        print(f"Q{question}: {count} complete runs, {ITERATIONS[question]} iterations each")
    return runs


def collect_files(root, runs):
    files = {}

    def include(source, archive_name):
        if not source.is_file():
            raise ValueError(f"Required file is missing: {source}")
        name = archive_name.as_posix()
        if name in files:
            raise ValueError(f"Duplicate archive path: {name}")
        files[name] = source

    for name in REQUIRED_FILES:
        include(root / name, Path(name))
    for source in sorted((root / "rob831").rglob("*.py")):
        if not excluded(source.relative_to(root)):
            include(source, source.relative_to(root))
    run_names = {run.name for run in runs}
    for run in runs:
        for source in sorted(run.rglob("*")):
            relative = source.relative_to(run)
            if source.is_file() and not excluded(relative):
                include(source, Path("run_logs") / run.name / relative)
    for question in ITERATIONS:
        result_dir = root / "results" / f"q{question}"
        summary = result_dir / "summary.csv"
        if not summary.is_file():
            raise ValueError(f"Required result summary is missing: {summary}")
        # Every plotted run must actually be included in the archive.
        with summary.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Result summary has no rows: {summary}")
        for row in rows:
            name = row.get("source_directory") or row.get("run_directory")
            if not name or Path(name).name not in run_names:
                raise ValueError(f"{summary}: summary refers to an unpackaged run: {name!r}")
        for source in sorted(result_dir.rglob("*")):
            relative = source.relative_to(root)
            if source.is_file() and source.suffix in RESULT_SUFFIXES and not excluded(relative):
                include(source, relative)
    return files


def write_archive(files, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate a temporary archive first, preserving any existing submit.zip if
    # input files change or validation fails while the archive is being written.
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, source in sorted(files.items()):
                archive.write(source, arcname=name)
        size = temporary.stat().st_size
        if size >= SIZE_LIMIT:
            raise ValueError(f"Archive is {size:,} bytes; it must be below {SIZE_LIMIT:,} bytes")
        with zipfile.ZipFile(temporary) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise ValueError(f"Archive CRC validation failed: {bad_file}")
            if set(archive.namelist()) != set(files):
                raise ValueError("Archive members do not match the validated file list")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {output}: {len(files)} files, {size:,} bytes; ZIP integrity verified")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "submit.zip")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without writing a ZIP")
    args = parser.parse_args()
    try:
        runs = complete_runs(args.data_dir)
        files = collect_files(root, runs)
        print(f"Validated {len(runs)} runs and {len(files)} archive inputs")
        if not args.check_only:
            write_archive(files, args.output)
    except (OSError, ValueError, ImportError, zipfile.BadZipFile) as error:
        print(f"Cannot package homework: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
