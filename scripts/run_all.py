#!/usr/bin/env python3
"""End-to-end: generate data -> train all conditions -> evaluate all -> print table."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared_matryoshka.evaluation.tables import format_console_table, format_latex_table
from shared_matryoshka.utils import setup_logging

PYTHON = sys.executable
SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPTS_DIR.parent / "configs"
RESULTS_DIR = Path("results")

# Conditions to run (in order)
CONDITIONS = [
    "v3_contrastive",
    "v3_mse",
    "v3_no_prefix",
    "single_dating",
    "single_hiring",
    "projection_heads",
    "adversarial",
]


def run_cmd(cmd: list[str], desc: str) -> None:
    log.info(f">>> {desc}")
    log.info(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    global log
    log = setup_logging()

    # Step 1: Generate data
    run_cmd(
        [PYTHON, str(SCRIPTS_DIR / "generate_data.py"), "--config", str(CONFIGS_DIR / "base.yaml")],
        "Generating data",
    )

    # Step 2: Train all conditions
    for condition in CONDITIONS:
        config_path = CONFIGS_DIR / f"{condition}.yaml"
        if not config_path.exists():
            log.warning(f"Config not found: {config_path}, skipping")
            continue
        run_cmd(
            [PYTHON, str(SCRIPTS_DIR / "train.py"), "--config", str(config_path)],
            f"Training {condition}",
        )

    # Step 3: Evaluate all conditions
    for condition in CONDITIONS:
        config_path = CONFIGS_DIR / f"{condition}.yaml"
        if not config_path.exists():
            continue
        run_cmd(
            [PYTHON, str(SCRIPTS_DIR / "evaluate.py"), "--config", str(config_path)],
            f"Evaluating {condition}",
        )

    # Step 4: Collect results and print table
    all_results = {}
    for condition in CONDITIONS:
        metrics_path = RESULTS_DIR / f"{condition}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_results[condition] = json.load(f)

    if all_results:
        log.info("\n" + "=" * 80)
        log.info("RESULTS SUMMARY")
        log.info("=" * 80 + "\n")
        print(format_console_table(all_results))
        print()

        # Save LaTeX table
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        latex = format_latex_table(all_results)
        with open(RESULTS_DIR / "results_table.tex", "w") as f:
            f.write(latex)
        log.info(f"\nLaTeX table saved to {RESULTS_DIR / 'results_table.tex'}")

        # Save combined results
        with open(RESULTS_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Combined results saved to {RESULTS_DIR / 'all_results.json'}")
    else:
        log.error("No results found!")


if __name__ == "__main__":
    main()
