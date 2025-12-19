# pdf_report.py

import os
import re
import textwrap
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from profiles import SurveyProfile, PROFILES


# -----------------------------
# Small helpers
# -----------------------------

YES_SET = {"YES", "Y", "TRUE", "1"}
NO_SET = {"NO", "N", "FALSE", "0"}


def _clean_series_as_str(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip()


def _get_unique_respondent_count(df_group: pd.DataFrame, name_idx: int) -> int:
    if name_idx < 0 or name_idx >= len(df_group.columns):
        return int(len(df_group))
    names = _clean_series_as_str(df_group.iloc[:, name_idx])
    names = names[names != ""]
    return int(names.nunique())


def _compose_group_title(profile: SurveyProfile, title_label: str, cycle_label: str) -> str:
    base = str(title_label).strip()

    if base == "All Teams":
        return f"All Teams - {cycle_label}"

    if " - " in base:
        return f"{base} - {cycle_label}"

    coach = (profile.team_coach_map or {}).get(base, "?")
    return f"{base} - {coach} - {cycle_label}"


def _format_count_pct_cell(profile: SurveyProfile, team_name: str, count: int) -> str:
    denom_map = profile.denominator_map
    if not denom_map:
        return str(count)

    total = denom_map.get(team_name)
    if not total or total <= 0:
        return str(count)

    pct = (count / float(total)) * 100.0
    return f"{count} ({pct:.0f}%)"


# -----------------------------
# Builders for low ratings / NO answers (generic, not players-only)
# -----------------------------

def build_low_ratings_table(
    df_group: pd.DataFrame,
    rating_indices: List[int],
    respondent_name_index: int,
    max_star: int = 3,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    out_cols: Dict[str, List[str]] = {}
    max_len = 0

    for idx in rating_indices:
        if idx < 0 or idx >= len(cols):
            continue

        q_name = str(cols[idx])
        series = pd.to_numeric(df_group.iloc[:, idx], errors="coerce")
        names = _clean_series_as_str(df_group.iloc[:, respondent_name_index])

        entries: List[str] = []
        for n, v in zip(names, series):
            if not n or pd.isna(v):
                continue
            try:
                rv = int(round(float(v)))
            except Exception:
                continue
            if 1 <= rv <= max_star:
                entries.append(f"{n}, ({rv}*)")

        # Keep column even if empty (we will still show header)
        out_cols[q_name] = entries
        max_len = max(max_len, len(entries))

    if not out_cols:
        return None

    if max_len == 0:
        for k in list(out_cols.keys()):
            out_cols[k] = [""]
        return pd.DataFrame(out_cols)

    for k in list(out_cols.keys()):
        vals = out_cols[k]
        out_cols[k] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(out_cols)


def build_no_answers_table(
    df_group: pd.DataFrame,
    yesno_indices: List[int],
    respondent_name_index: int,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    out_cols: Dict[str, List[str]] = {}
    max_len = 0

    for idx in yesno_indices:
        if idx < 0 or idx >= len(cols):
            continue

        q_name = str(cols[idx])
        series = _clean_series_as_str(df_group.iloc[:, idx]).str.upper()
        names = _clean_series_as_str(df_group.iloc[:, respondent_name_index])

        entries: List[str] = []
        for n, v in zip(names, series):
            if not n:
                continue
            if v in NO_SET:
                entries.append(n)

        out_cols[q_name] = entries
        max_len = max(max_len, len(entries))

    if not out_cols:
        return None

    if max_len == 0:
        for k in list(out_cols.keys()):
            out_cols[k] = [""]
        return pd.DataFrame(out_cols)

    for k in list(out_cols.keys()):
        vals = out_cols[k]
        out_cols[k] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(out_cols)


def _filter_low_df_by_max_star(low_df: pd.DataFrame, max_star: int = 2) -> pd.DataFrame:
    pattern = re.compile(r"\((\d)\*\)")
    new_cols: Dict[str, List[str]] = {}
    max_len = 0

    for col in low_df.columns:
        filtered: List[str] = []
        for val in low_df[col]:
            s = str(val).strip()
            if not s:
                continue
            m = pattern.search(s)
            if m:
                rating = int(m.group(1))
                if rating <= max_star:
                    filtered.append(s)
        new_cols[col] = filtered
        max_len = max(max_len, len(filtered))

    if max_len == 0:
        for col in new_cols:
            new_cols[col] = [""]
        return pd.DataFrame(new_cols)

    for col, vals in new_cols.items():
        new_cols[col] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(new_cols)


# -----------------------------
# Plot metadata
# -----------------------------

def _build_plot_metadata(profile: SurveyProfile, df_group: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = list(df_group.columns)

    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]
    yesno_indices = [i for i in (profile.yesno_col_indices or []) if i < len(cols)]
    has_choice = profile.choice_col_index is not None and profile.choice_col_index < len(cols)

    meta: List[Dict[str, Any]] = []
    number = 1

    for idx in rating_indices:
        meta.append({"ptype": "rating", "idx": idx, "col_name": cols[idx], "number": number})
        number += 1

    for idx in yesno_indices:
        meta.append({"ptype": "yesno", "idx": idx, "col_name": cols[idx], "number": number})
        number += 1

    if has_choice:
        meta.append(
            {
                "ptype": "choice",
                "idx": int(profile.choice_col_index),
                "col_name": cols[int(profile.choice_col_index)],
                "number": number,
            }
        )

    return meta


# -----------------------------
# Page: charts grid (page 1 per group)
# -----------------------------

def _add_group_charts_page_to_pdf(
    pdf: PdfPages,
    profile: SurveyProfile,
    df_group: pd.DataFrame,
    title_label: str,
    cycle_label: str,
    plots_meta: List[Dict[str, Any]],
) -> None:
    if not plots_meta:
        return

    n_resp = _get_unique_respondent_count(df_group, profile.respondent_name_index)
    noun = profile.respondent_singular if n_resp == 1 else profile.respondent_plural
    n_text = f" ({n_resp} {noun})"

    n_plots = len(plots_meta)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.5, 11))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    for ax, meta in zip(axes_flat, plots_meta):
        ptype = meta["ptype"]
        idx = meta["idx"]
        col_name = meta["col_name"]
        number = meta["number"]

        ax.text(
            0.02, 0.98, str(number),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
        )

        wrapped_name = textwrap.fill(str(col_name), width=40)

        if ptype == "rating":
            series = pd.to_numeric(df_group.iloc[:, idx], errors="coerce").dropna()
            counts = series.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            ax.bar(range(1, 6), counts.values)

            avg = series.mean() if not series.empty else None
            title = wrapped_name if avg is None or np.isnan(avg) else f"{wrapped_name}\n(Avg = {avg:.2f})"

            ax.set_title(title, fontsize=8)
            ax.set_xlabel("# of Stars", fontsize=7)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylim(0, max(counts.values.tolist() + [1]) * 1.2)

        elif ptype == "yesno":
            series = _clean_series_as_str(df_group.iloc[:, idx]).str.upper()
            yes_count = int(series.isin(YES_SET).sum())
            no_count = int(series.isin(NO_SET).sum())

            data = [yes_count, no_count]
            labels = ["YES", "NO"]

            if yes_count + no_count == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.axis("off")
            else:
                def make_label(pct, allvals=data):
                    total = sum(allvals)
                    count = int(round(pct * total / 100.0)) if total else 0
                    return f"{pct:.0f}%, {count}"

                ax.pie(data, labels=labels, autopct=make_label, textprops={"fontsize": 7})
                ax.set_title(wrapped_name, fontsize=8)

        elif ptype == "choice":
            series = df_group.iloc[:, idx].dropna().astype(str).str.strip()
            counts = series.value_counts()
            data = counts.values
            labels = counts.index.tolist()

            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.axis("off")
            else:
                def make_label(pct, allvals=data):
                    total = sum(allvals)
                    count = int(round(pct * total / 100.0)) if total else 0
                    return f"{pct:.0f}%, {count}"

                ax.pie(data, labels=labels, autopct=make_label, textprops={"fontsize": 7})
                ax.set_title(wrapped_name, fontsize=8)

    full_title = _compose_group_title(profile, title_label, cycle_label) + n_text
    fig.suptitle(full_title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# -----------------------------
# Respondent grid + comments
# -----------------------------

def _build_all_respondents_grid(
    df_group: pd.DataFrame,
    respondent_name_index: int,
    max_cols: int = 6,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    names = _clean_series_as_str(df_group.iloc[:, respondent_name_index])
    names = names[names != ""].drop_duplicates()

    if names.empty:
        return None

    n = len(names)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    grid = [["" for _ in range(ncols)] for _ in range(nrows)]
    i = 0
    for c in range(ncols):
        for r in range(nrows):
            if i >= n:
                break
            grid[r][c] = names.iloc[i]
            i += 1

    col_labels = [f"Respondents {i+1}" for i in range(ncols)]
    out = pd.DataFrame(grid, columns=col_labels)
    out = out[(out != "").any(axis=1)]
    return out


def _build_comments_table(
    df_group: pd.DataFrame,
    respondent_name_index: int,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    comment_indices: List[int] = []
    for i, name in enumerate(cols):
        nl = str(name).lower()
        if "comment" in nl or "suggest" in nl:
            comment_indices.append(i)

    if not comment_indices:
        return None

    rows: List[List[str]] = []
    for _, row in df_group.iterrows():
        who = str(row.iloc[respondent_name_index]).strip()
        if not who:
            continue

        for idx in comment_indices:
            val = row.iloc[idx]
            if pd.isna(val):
                continue
            txt = str(val).strip()
            if not txt:
                continue

            col_label = str(cols[idx])
            text_final = f"[{col_label}] {txt}" if len(comment_indices) > 1 else txt
            rows.append([who, text_final])

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["Respondent", "Comment / Suggestion"])


# -----------------------------
# Page: tables (page 2 per group)
# -----------------------------

def _add_group_tables_page_to_pdf(
    pdf: PdfPages,
    profile: SurveyProfile,
    df_group: pd.DataFrame,
    title_label: str,
    cycle_label: str,
    plots_meta: List[Dict[str, Any]],
    is_all_teams: bool,
) -> None:
    n_resp = _get_unique_respondent_count(df_group, profile.respondent_name_index)
    noun = profile.respondent_singular if n_resp == 1 else profile.respondent_plural
    n_text = f" ({n_resp} {noun})"

    rating_indices = [m["idx"] for m in plots_meta if m["ptype"] == "rating"]
    yesno_indices = [m["idx"] for m in plots_meta if m["ptype"] == "yesno"]

    rating_number_by_name = {m["col_name"]: m["number"] for m in plots_meta if m["ptype"] == "rating"}
    yesno_number_by_name = {m["col_name"]: m["number"] for m in plots_meta if m["ptype"] == "yesno"}

    low_df = None
    low_labels = None
    if rating_indices:
        low_df = build_low_ratings_table(
            df_group,
            rating_indices=rating_indices,
            respondent_name_index=profile.respondent_name_index,
            max_star=3,
        )
        if low_df is not None and is_all_teams:
            low_df = _filter_low_df_by_max_star(low_df, max_star=2)

        if low_df is not None:
            low_labels = []
            for col in low_df.columns:
                num = rating_number_by_name.get(col)
                if num is not None and profile.chart_labels and num in profile.chart_labels:
                    low_labels.append(profile.chart_labels[num])
                else:
                    low_labels.append(str(col))

    no_df = None
    no_labels = None
    if yesno_indices:
        no_df = build_no_answers_table(
            df_group,
            yesno_indices=yesno_indices,
            respondent_name_index=profile.respondent_name_index,
        )
        if no_df is not None:
            no_labels = []
            for col in no_df.columns:
                num = yesno_number_by_name.get(col)
                if num is not None and profile.chart_labels and num in profile.chart_labels:
                    no_labels.append(profile.chart_labels[num])
                else:
                    no_labels.append(str(col))

    completion_df = None
    respondents_df = None
    comments_df = None

    if is_all_teams:
        completion_df = pd.DataFrame(
            {"Metric": [f"{profile.respondent_plural.capitalize()} who completed this survey"], "Value": [n_resp]}
        )
    else:
        respondents_df = _build_all_respondents_grid(
            df_group,
            respondent_name_index=profile.respondent_name_index,
            max_cols=6,
        )
        comments_df = _build_comments_table(df_group, respondent_name_index=profile.respondent_name_index)

    if (
        low_df is None and
        no_df is None and
        completion_df is None and
        respondents_df is None and
        comments_df is None
    ):
        return

    sections: List[str] = []
    if low_df is not None:
        sections.append("low")
    if no_df is not None:
        sections.append("no")
    if is_all_teams:
        if completion_df is not None:
            sections.append("completion")
    else:
        if respondents_df is not None:
            sections.append("respondents")
        if comments_df is not None:
            sections.append("comments")

    height_ratios: List[float] = []
    for s in sections:
        if s == "low":
            height_ratios.append(1.1)
        elif s == "no":
            height_ratios.append(0.9)
        elif s == "completion":
            height_ratios.append(0.8)
        elif s == "respondents":
            height_ratios.append(1.3)
        elif s == "comments":
            height_ratios.append(1.6)

    fig, axes = plt.subplots(
        nrows=len(sections),
        ncols=1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(sections) == 1:
        axes = [axes]

    row_idx = 0

    if low_df is not None:
        ax = axes[row_idx]
        ax.axis("off")
        table = ax.table(
            cellText=low_df.values,
            colLabels=low_labels if low_labels is not None else low_df.columns,
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        ncols_low = len(low_df.columns)
        width_scale = 1.0 if ncols_low <= 8 else (0.85 if ncols_low <= 12 else 0.7)
        table.scale(width_scale, 1.15)
        ax.set_title("1-2 Star Reviews (columns = chart numbers)" if is_all_teams else "1-3 Star Reviews (columns = chart numbers)", fontsize=10, pad=6)
        row_idx += 1

    if no_df is not None:
        ax = axes[row_idx]
        ax.axis("off")
        table = ax.table(
            cellText=no_df.values,
            colLabels=no_labels if no_labels is not None else no_df.columns,
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        ncols_no = len(no_df.columns)
        width_scale = 1.0 if ncols_no <= 6 else (0.9 if ncols_no <= 10 else 0.75)
        table.scale(width_scale, 1.15)
        ax.set_title('"NO" Replies (columns = chart numbers)', fontsize=10, pad=6)
        row_idx += 1

    if is_all_teams and completion_df is not None:
        ax = axes[row_idx]
        ax.axis("off")
        table = ax.table(
            cellText=completion_df.values,
            colLabels=completion_df.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        ax.set_title("Survey completion summary", fontsize=10, pad=4)
        row_idx += 1

    if (not is_all_teams) and (respondents_df is not None):
        ax = axes[row_idx]
        ax.axis("off")
        table = ax.table(
            cellText=respondents_df.values,
            colLabels=respondents_df.columns,
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.6)  # make cells taller
        ax.set_title(f"{profile.respondent_plural.capitalize()} who completed this survey", fontsize=10, pad=4)
        row_idx += 1

    if (not is_all_teams) and (comments_df is not None):
        ax = axes[row_idx]
        ax.axis("off")
        table = ax.table(
            cellText=comments_df.values,
            colLabels=comments_df.columns,
            loc="upper left",
            colWidths=[0.12, 0.88],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Taller rows. If you want even taller, bump 2.2 -> 2.6
        table.scale(1.05, 2.2)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(ha="center", va="center", fontweight="bold")
                continue
            if c == 0:
                cell.set_text_props(ha="center", va="top")
            else:
                txt = cell.get_text()
                txt.set_wrap(True)
                txt.set_ha("left")
                txt.set_va("top")
                cell.PAD = 0.02

        ax.set_title("Comments and Suggestions", fontsize=10, pad=6)

    full_title = _compose_group_title(profile, title_label, cycle_label) + n_text + " (Details)"
    fig.suptitle(full_title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.subplots_adjust(hspace=0.55)
    pdf.savefig(fig)
    plt.close(fig)


# -----------------------------
# Page: cycle summary (page 1)
# -----------------------------

def _add_cycle_summary_page(
    pdf: PdfPages,
    profile: SurveyProfile,
    df: pd.DataFrame,
    cycle_label: str,
) -> None:
    cols = list(df.columns)
    if profile.group_col_index >= len(cols):
        raise ValueError("Group column index is outside the available columns.")

    group_col_name = df.columns[profile.group_col_index]

    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]

    # Pick QQ rating index:
    # - if profile explicitly sets qq_rating_col_index, use it
    # - else mimic your old logic: 7th rating question (index 6) if present
    if profile.qq_rating_col_index is not None and profile.qq_rating_col_index < len(cols):
        qq_idx = int(profile.qq_rating_col_index)
    else:
        qq_idx = rating_indices[6] if len(rating_indices) >= 7 else None

    stats_by_team: Dict[str, Tuple[int, float]] = {}

    for group_value, group_df in df.groupby(group_col_name, sort=True):
        team_name = str(group_value).strip()
        if team_name == profile.unassigned_label:
            continue

        n_resp = _get_unique_respondent_count(group_df, profile.respondent_name_index)

        if qq_idx is not None and qq_idx < len(group_df.columns):
            series = pd.to_numeric(group_df.iloc[:, qq_idx], errors="coerce").dropna()
            avg_rating = float(series.mean()) if not series.empty else np.nan
        else:
            avg_rating = np.nan

        stats_by_team[team_name] = (n_resp, avg_rating)

    all_team_names = sorted(set(stats_by_team.keys()) | set((profile.team_coach_map or {}).keys()))
    if not all_team_names:
        return

    rows: List[Dict[str, Any]] = []
    for team_name in all_team_names:
        coach = (profile.team_coach_map or {}).get(team_name, "?")
        count, avg_rating = stats_by_team.get(team_name, (0, np.nan))
        rows.append({"Team": team_name, "Coach": coach, "Count": count, "Rating": avg_rating})

    summary_df = pd.DataFrame(rows)

    total_responses = int(summary_df["Count"].sum())
    total_str = f"{total_responses} {profile.respondent_singular}" if total_responses == 1 else f"{total_responses} {profile.respondent_plural}"

    # Sort (same idea as your old one)
    summary_df = summary_df.sort_values(by=["Count", "Rating"], ascending=[False, False], ignore_index=True)

    summary_df["TeamCoach"] = summary_df["Team"] + " - " + summary_df["Coach"]
    summary_df["RatingStr"] = summary_df["Rating"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    summary_df["CountDisplay"] = [
        _format_count_pct_cell(profile, team, int(c))
        for team, c in zip(summary_df["Team"], summary_df["Count"])
    ]

    # Compute QQ index = rating * completion fraction
    qq_vals: List[float] = []
    for team, count, rating in zip(summary_df["Team"], summary_df["Count"], summary_df["Rating"]):
        if pd.isna(rating):
            qq_vals.append(0.0)
            continue

        denom_map = profile.denominator_map
        if denom_map and denom_map.get(team) and denom_map.get(team) > 0:
            frac = float(count) / float(denom_map[team])
        else:
            # If no denominator, treat completion fraction as 1.0 so QQ = rating
            frac = 1.0

        qq_vals.append(float(rating) * frac)

    summary_df["QQIndex"] = qq_vals

    fig, (ax_table, ax_bar) = plt.subplots(
        1, 2,
        figsize=(11, 8.5),
        gridspec_kw={"width_ratios": [1.2, 1.8]},
    )
    fig.suptitle(f"{cycle_label} Summary", fontsize=14, fontweight="bold")

    ax_table.axis("off")
    display_df = summary_df[["TeamCoach", "CountDisplay", "RatingStr"]]

    table = ax_table.table(
        cellText=display_df.values,
        colLabels=["Team - Coach", profile.respondent_plural.capitalize(), "Rating"],
        loc="center",
        colWidths=[0.72, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.1, 1.2)  # slightly taller cells

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(ha="center", va="center", fontweight="bold")
        else:
            cell.set_text_props(ha=("left" if c == 0 else "center"), va="center")

    ax_bar.set_title(f"{cycle_label} QQ (Quality-Quantity) Index - {total_str}", fontsize=10)

    y_pos = np.arange(len(summary_df))
    ax_bar.barh(y_pos, summary_df["QQIndex"].values.astype(float), height=0.6, label="QQ index")
    ax_bar.set_xlabel("QQ index (rating * completion fraction)")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(summary_df["TeamCoach"], fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 5.1)
    ax_bar.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# -----------------------------
# Main entry (combined)
# -----------------------------

def create_pdf_report(
    input_path: str,
    cycle_label: str = "Cycle",
    survey_type: str = "players",
    output_path: Optional[str] = None,
) -> str:
    profile = PROFILES.get(survey_type.lower().strip())
    if profile is None:
        raise ValueError(f"Unknown survey_type: {survey_type}. Use one of: {list(PROFILES.keys())}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + f"_{profile.key}_report.pdf"

    df = pd.read_excel(input_path, sheet_name=0)

    if profile.group_col_index >= len(df.columns):
        raise ValueError("Group column is outside the available columns in the sheet.")

    group_col_name = df.columns[profile.group_col_index]
    df[group_col_name] = df[group_col_name].fillna(profile.unassigned_label)

    # Compute QQ order for teams with data
    cols = list(df.columns)
    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]

    if profile.qq_rating_col_index is not None and profile.qq_rating_col_index < len(cols):
        qq_idx = int(profile.qq_rating_col_index)
    else:
        qq_idx = rating_indices[6] if len(rating_indices) >= 7 else None

    stats_rows: List[Dict[str, Any]] = []
    for g, group_df in df.groupby(group_col_name, sort=False):
        team = str(g).strip()
        if team == profile.unassigned_label:
            continue

        count = _get_unique_respondent_count(group_df, profile.respondent_name_index)

        if qq_idx is not None and qq_idx < len(group_df.columns):
            series = pd.to_numeric(group_df.iloc[:, qq_idx], errors="coerce").dropna()
            avg = float(series.mean()) if not series.empty else np.nan
        else:
            avg = np.nan

        stats_rows.append({"Team": team, "Count": count, "Avg": avg})

    if not stats_rows:
        # No data; still return path (empty PDF not generated)
        return output_path

    stats_df = pd.DataFrame(stats_rows)

    qq_vals: List[float] = []
    for team, count, avg in zip(stats_df["Team"], stats_df["Count"], stats_df["Avg"]):
        if pd.isna(avg):
            qq_vals.append(0.0)
            continue

        denom_map = profile.denominator_map
        if denom_map and denom_map.get(team) and denom_map.get(team) > 0:
            frac = float(count) / float(denom_map[team])
        else:
            frac = 1.0

        qq_vals.append(float(avg) * frac)

    stats_df["QQIndex"] = qq_vals
    stats_df = stats_df.sort_values("QQIndex", ascending=False, ignore_index=True)
    qq_sorted_teams = list(stats_df["Team"].values)

    grouped: Dict[str, pd.DataFrame] = {
        str(g).strip(): sub_df for g, sub_df in df.groupby(group_col_name, sort=False)
    }

    with PdfPages(output_path) as pdf:
        _add_cycle_summary_page(pdf, profile, df, cycle_label)

        # All teams pages
        all_meta = _build_plot_metadata(profile, df)
        _add_group_charts_page_to_pdf(pdf, profile, df, "All Teams", cycle_label, all_meta)
        _add_group_tables_page_to_pdf(pdf, profile, df, "All Teams", cycle_label, all_meta, is_all_teams=True)

        # Team pages in QQ order
        for team in qq_sorted_teams:
            group_df = grouped.get(team)
            if group_df is None:
                continue

            coach = (profile.team_coach_map or {}).get(team, "?")
            title_label = f"{team} - {coach}"

            meta = _build_plot_metadata(profile, group_df)
            _add_group_charts_page_to_pdf(pdf, profile, group_df, title_label, cycle_label, meta)
            _add_group_tables_page_to_pdf(pdf, profile, group_df, title_label, cycle_label, meta, is_all_teams=False)

    return output_path


# Backward-compatible name if you want:
def create_pdf_from_original(input_path: str, cycle_label: str = "Cycle", output_path: Optional[str] = None) -> str:
    return create_pdf_report(input_path=input_path, cycle_label=cycle_label, survey_type="players", output_path=output_path)
