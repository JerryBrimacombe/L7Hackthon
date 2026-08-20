"""Generate a maintained EDA artifact from the v006/v0083 datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import config
from .profiles import POLICIES, build_profile


def _symptom_positive_rate_table(frame):
    rows = []
    for feature in ["Cough", "Fever", "Sore_throat", "Shortness_of_breath", "Headache"]:
        subset = frame[frame[feature] == 1]
        rate = float(subset["label"].mean()) if len(subset) else float("nan")
        rows.append((feature, len(subset), rate))
    return rows


def build_markdown(version: str, missing_policy: str) -> str:
    frame = build_profile(version=version, missing_policy=missing_policy)
    prevalence = float(frame["label"].mean())
    symptom_rows = _symptom_positive_rate_table(frame)

    lines = [
        "# EDA Summary",
        "",
        f"- Dataset version: `{version}`",
        f"- Missing policy: `{missing_policy}`",
        f"- Rows: `{len(frame)}`",
        f"- Positive rate: `{prevalence:.2%}`",
        "",
        "## Symptom prevalence and positivity",
        "",
        "| Feature | Rows with symptom=1 | Positive rate within symptom |",
        "| --- | ---: | ---: |",
    ]
    for feature, count, rate in symptom_rows:
        lines.append(f"| {feature} | {count} | {rate:.2%} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This artifact is generated from code, not handwritten notebook cells.",
            "- Use it as a reproducible narrative snapshot before model comparisons.",
        ]
    )
    return "\n".join(lines)


def render_symptom_chart(version: str, missing_policy: str, out_path: Path) -> None:
    frame = build_profile(version=version, missing_policy=missing_policy)
    rows = _symptom_positive_rate_table(frame)
    names = [r[0] for r in rows]
    rates = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(names, rates)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(positive | symptom = 1)")
    ax.set_title("Symptom-specific positivity rate")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate maintained EDA report artifacts.")
    parser.add_argument("--dataset", default="v006", choices=list(config.DATASETS))
    parser.add_argument("--missing-policy", default="paper", choices=list(POLICIES))
    args = parser.parse_args()

    out_dir = config.DOCS_DIR / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"eda_{args.dataset}_{args.missing_policy}.md"
    md_path.write_text(
        build_markdown(version=args.dataset, missing_policy=args.missing_policy),
        encoding="utf-8",
    )

    chart_path = out_dir / f"eda_{args.dataset}_{args.missing_policy}_symptom_rates.png"
    render_symptom_chart(args.dataset, args.missing_policy, chart_path)

    print(f"Wrote {md_path}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()