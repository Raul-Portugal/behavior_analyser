"""Render a :class:`core.validation_report.ValidationResults` as one HTML file.

The page is self-contained: inline CSS (light and dark), plots embedded as
base64 PNG (drawn with matplotlib's ``Figure`` and ``FigureCanvasAgg``, no
pyplot state), and no external fonts, scripts or links. Every table is also
written as a CSV file next to the report by :func:`write_report`.
"""
import base64
import html
import io
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.agreement_stats import BlandAltmanResult
from core.manual_scoring import PARADIGM_LABELS
from core.validation_report import (ALL_ROLES, PARADIGM_ORDER, MetricAgreement,
                                    Table, ValidationResults, build_tables,
                                    write_table_csv)
from core.version import APP_NAME

# One accent hue (points, bars) plus neutral ink; chosen to stay legible in
# print and for colour-blind readers because nothing relies on hue alone.
ACCENT = "#2a5db0"
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
WARN = "#b3261e"


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _new_axes(width: float = 4.6, height: float = 3.9):
    figure = Figure(figsize=(width, height), dpi=110, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_facecolor("white")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=INK, labelsize=8)
    axes.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    return figure, axes


def _png_base64(figure: Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def scatter_plot(agreement: MetricAgreement) -> str:
    """Automated against manual, with the identity line."""
    figure, axes = _new_axes()
    pairs = np.array(
        [(a, m) for a, m in agreement.pairs
         if math.isfinite(a) and math.isfinite(m)], dtype=float)
    unit = f" ({agreement.unit})" if agreement.unit else ""
    if pairs.size:
        low = float(pairs.min())
        high = float(pairs.max())
        pad = (high - low) * 0.06 or 1.0
        low, high = low - pad, high + pad
        axes.plot([low, high], [low, high], color=MUTED, linestyle="--",
                  linewidth=1.0, label="identity")
        axes.scatter(pairs[:, 1], pairs[:, 0], s=26, color=ACCENT,
                     edgecolor="white", linewidth=0.8, zorder=3)
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.set_aspect("equal", adjustable="box")
    icc = agreement.summary.icc
    caption = f"n = {agreement.summary.n}"
    if icc is not None and math.isfinite(icc.icc):
        caption += f"   ICC(2,1) = {icc.icc:.3f}"
    axes.set_title(f"{agreement.label}: automated vs manual", fontsize=9.5,
                   color=INK, loc="left")
    axes.set_xlabel(f"Manual{unit}", fontsize=8.5, color=INK)
    axes.set_ylabel(f"Automated{unit}", fontsize=8.5, color=INK)
    axes.text(0.03, 0.97, caption, transform=axes.transAxes, fontsize=8,
              va="top", color=INK)
    figure.tight_layout()
    return _png_base64(figure)


def bland_altman_plot(agreement: MetricAgreement) -> str:
    """Difference (automated - manual) against the pair mean."""
    ba: BlandAltmanResult = agreement.summary.bland_altman
    figure, axes = _new_axes(5.0, 3.9)
    unit = f" ({agreement.unit})" if agreement.unit else ""
    if ba.n:
        axes.scatter(ba.means, ba.differences, s=26, color=ACCENT,
                     edgecolor="white", linewidth=0.8, zorder=3)
        axes.axhline(0, color=GRID, linewidth=1.0, zorder=1)
        axes.axhline(ba.bias, color=INK, linewidth=1.4)
        axes.text(1.0, ba.bias, f" bias {ba.bias:+.3g}",
                  transform=axes.get_yaxis_transform(), fontsize=7.5,
                  va="center", color=INK)
        if math.isfinite(ba.loa_lower):
            for value, name in ((ba.loa_upper, "upper"),
                                (ba.loa_lower, "lower")):
                axes.axhline(value, color=MUTED, linestyle="--", linewidth=1.1)
                axes.text(1.0, value, f" {value:+.3g}", transform=axes.get_yaxis_transform(),
                          fontsize=7.5, va="center", color=MUTED)
    axes.set_title(f"{agreement.label}: Bland-Altman", fontsize=9.5,
                   color=INK, loc="left")
    axes.set_xlabel(f"Mean of automated and manual{unit}", fontsize=8.5,
                    color=INK)
    axes.set_ylabel(f"Automated - manual{unit}", fontsize=8.5, color=INK)
    figure.tight_layout()
    return _png_base64(figure)


def confusion_plot(title: str, categories: Sequence[str],
                   matrix: np.ndarray) -> str:
    """Counts of automated (rows) against manual (columns) labels."""
    size = max(4.2, 1.0 + 0.95 * len(categories))
    figure, axes = _new_axes(size, size * 0.92)
    axes.grid(False)
    shown = np.asarray(matrix, dtype=float)
    axes.imshow(shown, cmap="Blues", vmin=0, aspect="auto")
    axes.set_xticks(range(len(categories)))
    axes.set_yticks(range(len(categories)))
    axes.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
    axes.set_yticklabels(categories, fontsize=8)
    peak = shown.max() if shown.size else 0.0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            value = int(shown[i, j])
            axes.text(j, i, f"{value}", ha="center", va="center", fontsize=8,
                      color="white" if peak and shown[i, j] > peak / 2 else INK)
    axes.set_xlabel("Manual", fontsize=8.5, color=INK)
    axes.set_ylabel("Automated", fontsize=8.5, color=INK)
    axes.set_title(title, fontsize=9.5, color=INK, loc="left")
    for side in ("left", "bottom"):
        axes.spines[side].set_visible(False)
    figure.tight_layout()
    return _png_base64(figure)


def error_histogram(errors_px: np.ndarray) -> str:
    figure, axes = _new_axes()
    if errors_px.size:
        axes.hist(errors_px, bins=min(30, max(5, errors_px.size // 2)),
                  color=ACCENT, edgecolor="white", linewidth=0.6)
        axes.axvline(float(np.median(errors_px)), color=INK, linewidth=1.3,
                     label=f"median {np.median(errors_px):.2f} px")
        axes.legend(loc="upper right", fontsize=7.5, frameon=False)
    axes.set_title("Position error (matched frames)", fontsize=9.5,
                   color=INK, loc="left")
    axes.set_xlabel("Distance between tracker and click (px)", fontsize=8.5,
                    color=INK)
    axes.set_ylabel("Frames", fontsize=8.5, color=INK)
    figure.tight_layout()
    return _png_base64(figure)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root { --bg:#ffffff; --ink:#222222; --muted:#6b6b6b; --line:#d8d8d8;
  --head:#f1f3f6; --accent:#2a5db0; --pass:#1b6e3a; --fail:#b3261e;
  --na:#6b6b6b; --pass-bg:#e4f3ea; --fail-bg:#fbe7e5; --na-bg:#eeeeee; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181c; --ink:#e6e6e6; --muted:#a0a0a0; --line:#3a3d44;
    --head:#23262c; --accent:#8fb3f0; --pass:#7fd39c; --fail:#ff9a92;
    --na:#a0a0a0; --pass-bg:#173523; --fail-bg:#40201e; --na-bg:#2a2c31; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 36px 0 8px; padding-top: 12px;
  border-top: 1px solid var(--line); }
h3 { font-size: 1.02rem; margin: 22px 0 6px; }
p, li { max-width: 78ch; }
.meta, .note { color: var(--muted); font-size: 0.9rem; }
.table-wrap { overflow-x: auto; margin: 6px 0 14px; }
table { border-collapse: collapse; font-size: 0.86rem; min-width: 60%; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: right;
  white-space: nowrap; }
th { background: var(--head); font-weight: 600; text-align: center; }
td.text, th.text { text-align: left; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-weight: 600; font-size: 0.8rem; }
.pass { color: var(--pass); background: var(--pass-bg); }
.fail { color: var(--fail); background: var(--fail-bg); }
.na { color: var(--na); background: var(--na-bg); }
.plots { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 18px; }
.plots figure { margin: 0; }
.plots img { max-width: 100%; height: auto; border: 1px solid var(--line);
  background: #ffffff; }
figcaption { color: var(--muted); font-size: 0.8rem; }
.warnings li { color: var(--fail); }
code { font-size: 0.85em; word-break: break-all; }
"""


def _fmt(value) -> str:
    if value is None:
        return "&ndash;"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return "&ndash;"
        if abs(value) >= 1000:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return html.escape(str(value))


def _is_text(value) -> bool:
    return isinstance(value, str)


def table_html(table: Table, with_title: bool = True) -> str:
    """One table as HTML (numbers right-aligned, text left-aligned)."""
    head = "".join(
        f"<th class='text'>{html.escape(str(c))}</th>" for c in table.columns)
    body = []
    for row in table.rows:
        cells = []
        for cell in row:
            css = " class='text'" if _is_text(cell) else ""
            if _is_text(cell) and cell in ("pass", "fail", "not evaluable"):
                css_name = {"pass": "pass", "fail": "fail",
                            "not evaluable": "na"}[cell]
                label = {"pass": "PASS", "fail": "FAIL",
                         "not evaluable": "NOT EVALUABLE"}[cell]
                cells.append(
                    f"<td class='text'><span class='badge {css_name}'>"
                    f"{label}</span></td>")
            else:
                cells.append(f"<td{css}>{_fmt(cell)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(
            f"<tr><td class='text' colspan='{max(1, len(table.columns))}'>"
            "No rows.</td></tr>")
    title = f"<h3>{html.escape(table.title)}</h3>" if with_title else ""
    note = f"<p class='note'>{html.escape(table.note)}</p>" if table.note else ""
    return (f"{title}<div class='table-wrap'><table><thead><tr>{head}</tr>"
            f"</thead><tbody>{''.join(body)}</tbody></table></div>{note}")


def _figure(png: str, caption: str, alt: str) -> str:
    return (f"<figure><img alt='{html.escape(alt)}' "
            f"src='data:image/png;base64,{png}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


_PER_VIDEO_TABLE = {
    "y_maze": "y_maze_per_video", "epm": "epm_per_video",
    "open_field": "open_field_per_video", "tst": "tst_per_mouse",
}


def render_html(results: ValidationResults) -> str:
    """The complete report page."""
    tables: Dict[str, Table] = {t.name: t for t in build_tables(results)}
    config = results.config
    parts: List[str] = []

    criteria = results.criteria
    passed = sum(r.status == "pass" for r in criteria)
    failed = sum(r.status == "fail" for r in criteria)
    unknown = sum(r.status == "not evaluable" for r in criteria)

    parts.append(
        f"<h1>{html.escape(config.title)}</h1>"
        f"<p class='meta'>{html.escape(APP_NAME)} {html.escape(results.app_version)}"
        f" &middot; generated {html.escape(results.generated)} &middot; "
        f"event tolerance {config.tolerance_s:g} s &middot; "
        f"{config.confidence * 100:g}% confidence intervals</p>")
    if config.notes.strip():
        parts.append(f"<p>{html.escape(config.notes.strip())}</p>")
    parts.append(
        "<p>Agreement between automated scoring and manual scoring. "
        "Differences are automated minus manual. Statistics come from the "
        "validation module distributed with the software; see "
        "VALIDATION_PROTOCOL.md for how to choose the validation set and "
        "interpret the numbers.</p>")

    parts.append("<h2>Acceptance criteria</h2>")
    if criteria:
        parts.append(
            f"<p><span class='badge pass'>{passed} PASS</span> "
            f"<span class='badge fail'>{failed} FAIL</span> "
            f"<span class='badge na'>{unknown} NOT EVALUABLE</span></p>")
        parts.append(table_html(tables["acceptance_criteria"], False))
        parts.append(
            "<p class='note'>Criteria are the thresholds entered before "
            "scoring. A criterion is not evaluable when its statistic could "
            "not be computed (no data, or fewer than two videos).</p>")
    else:
        parts.append("<p>No acceptance criteria were entered.</p>")

    parts.append("<h2>Pooled agreement</h2>")
    parts.append(table_html(tables["pooled_agreement"], False))
    for name in ("pooled_events", "frame_agreement"):
        if name in tables:
            parts.append(table_html(tables[name]))

    for paradigm in PARADIGM_ORDER:
        result = results.paradigms.get(paradigm)
        if result is None:
            continue
        parts.append(f"<h2>{html.escape(PARADIGM_LABELS[paradigm])}</h2>")
        for table in tables.values():
            if table.name in ("y_maze_per_video", "epm_per_video",
                              "open_field_per_video", "tst_per_mouse") \
                    and table.columns == result.columns:
                parts.append(table_html(table))
        for metric, agreement in result.agreements.items():
            if agreement.summary.n < 1:
                continue
            alt = f"{agreement.label} ({agreement.unit})"
            parts.append(f"<h3>{html.escape(alt)}</h3><div class='plots'>")
            parts.append(_figure(
                bland_altman_plot(agreement),
                f"Bland-Altman, n = {agreement.summary.n}",
                f"Bland-Altman plot of {alt}"))
            parts.append(_figure(
                scatter_plot(agreement), "Dashed line: perfect agreement",
                f"Scatter plot of automated against manual {alt}"))
            parts.append("</div>")
        for confusion in result.confusions:
            parts.append("<div class='plots'>")
            kappa = confusion.kappa
            caption = f"{kappa.n} frames from {confusion.n_videos} video(s); " + (
                f"kappa {kappa.kappa:.3f}" if math.isfinite(kappa.kappa)
                else "kappa undefined")
            parts.append(_figure(
                confusion_plot(confusion.title, confusion.categories,
                               confusion.matrix),
                caption, f"Confusion matrix: {confusion.title}"))
            parts.append("</div>")
            parts.append(table_html(
                tables[f"{confusion.name}_confusion"]))

    if results.spotcheck is not None:
        parts.append("<h2>Tracking position spot-check</h2>")
        parts.append(table_html(tables["position_spotcheck_pooled"], False))
        parts.append(table_html(tables["position_spotcheck_per_video"]))
        if results.spotcheck.errors_px.size:
            parts.append("<div class='plots'>" + _figure(
                error_histogram(results.spotcheck.errors_px),
                f"{results.spotcheck.errors_px.size} matched frames",
                "Histogram of position errors") + "</div>")

    if results.warnings:
        parts.append("<h2>Warnings</h2><ul class='warnings'>"
                     + "".join(f"<li>{html.escape(w)}</li>"
                               for w in results.warnings) + "</ul>")

    parts.append("<h2>Reproducibility</h2>")
    parts.append(
        f"<p>Application: {html.escape(APP_NAME)} "
        f"{html.escape(results.app_version)}. Report generated "
        f"{html.escape(results.generated)}. Event tolerance "
        f"{config.tolerance_s:g} s; confidence level "
        f"{config.confidence * 100:g}%.</p>")
    parts.append(table_html(tables["inputs"]))
    parts.append(table_html(tables["input_files"]))
    parts.append("<p class='note'>Roles: " + ", ".join(ALL_ROLES) + ".</p>")

    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(config.title)}</title><style>{_CSS}</style>"
        f"</head><body><main>{''.join(parts)}</main></body></html>\n")


def write_report(
    results: ValidationResults, output_dir, name: str = "validation_report"
) -> Tuple[Path, List[Path]]:
    """Write ``<name>.html`` and ``<name>_<table>.csv`` for every table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Path] = []
    for table in build_tables(results):
        path = output_dir / f"{name}_{table.name}.csv"
        write_table_csv(path, table)
        csv_paths.append(path)
    html_path = output_dir / f"{name}.html"
    html_path.write_text(render_html(results), encoding="utf-8")
    return html_path, csv_paths


__all__ = ["bland_altman_plot", "confusion_plot", "error_histogram",
           "render_html", "scatter_plot", "table_html", "write_report"]
