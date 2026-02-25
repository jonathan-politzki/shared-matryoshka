"""Console + LaTeX table formatting for results."""

from __future__ import annotations

from tabulate import tabulate


# Key metrics to show in the summary table
SUMMARY_METRICS = [
    "cross_domain_accuracy",
    "cross_domain_margin",
    "recall@1",
    "recall@5",
    "mrr",
    "dating_triplet_accuracy",
    "hiring_triplet_accuracy",
    "prefix_variance_dating",
]

SHORT_NAMES = {
    "cross_domain_accuracy": "XD Acc",
    "cross_domain_margin": "XD Margin",
    "recall@1": "R@1",
    "recall@5": "R@5",
    "mrr": "MRR",
    "dating_triplet_accuracy": "Dating Acc",
    "hiring_triplet_accuracy": "Hiring Acc",
    "prefix_variance_dating": "Prefix Var",
}


def format_console_table(all_results: dict[str, dict[str, float]]) -> str:
    """Format results as a console table.

    Args:
        all_results: {method_name: {metric: value}}.
    """
    headers = ["Method"] + [SHORT_NAMES.get(m, m) for m in SUMMARY_METRICS]
    rows = []
    for method, res in sorted(all_results.items()):
        row = [method]
        for m in SUMMARY_METRICS:
            val = res.get(m, float("nan"))
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def format_latex_table(all_results: dict[str, dict[str, float]]) -> str:
    """Format results as a LaTeX table."""
    metrics = SUMMARY_METRICS
    col_headers = [SHORT_NAMES.get(m, m) for m in metrics]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of methods across evaluation metrics.}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(col_headers) + " \\\\")
    lines.append("\\midrule")

    for method, res in sorted(all_results.items()):
        vals = []
        for m in metrics:
            v = res.get(m, float("nan"))
            vals.append(f"{v:.3f}")
        lines.append(f"{method} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)
