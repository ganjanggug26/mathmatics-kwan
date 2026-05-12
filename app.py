from __future__ import annotations

import math
import re
import base64
from io import BytesIO
from html import escape
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)


x = sp.symbols("x")
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class AnalysisResult:
    expr: sp.Expr
    first_derivative: sp.Expr
    second_derivative: sp.Expr
    x_intercepts: list[sp.Expr]
    y_intercept: float | None
    critical_points: list[sp.Expr]
    inflection_candidates: list[sp.Expr]
    vertical_asymptotes: list[sp.Expr]


def parse_function(raw_expression: str) -> sp.Expr:
    cleaned = raw_expression.strip()
    if cleaned.startswith("y="):
        cleaned = cleaned[2:].strip()
    if cleaned.startswith("f(x)="):
        cleaned = cleaned[5:].strip()
    return sp.simplify(
        parse_expr(
            cleaned,
            local_dict={"x": x, "e": sp.E, "pi": sp.pi, "ln": sp.log, "log": sp.log},
            transformations=TRANSFORMATIONS,
        )
    )


def expr_to_float(value: sp.Expr | float | int) -> float:
    return float(sp.N(sp.re(value)))


def pretty_text(value: sp.Expr | float | int) -> str:
    text = sp.sstr(sp.nsimplify(value))
    text = re.sub(r"sqrt\(([^()]+)\)", r"√\1", text)
    return text.replace("**", "^")


def exact_text(value: sp.Expr | float | int) -> str:
    return pretty_text(value)


def pretty_html(value: sp.Expr | float | int | str) -> str:
    text = escape(str(value).replace("**", "^"))
    text = re.sub(
        r"sqrt\(([^()]+)\)",
        lambda match: (
            "<span class='sqrt-expr'>"
            "<span class='sqrt-symbol'>√</span>"
            f"<span class='radicand'>{escape(match.group(1))}</span>"
            "</span>"
        ),
        text,
    )
    text = re.sub(
        r"√([A-Za-z0-9]+)",
        lambda match: (
            "<span class='sqrt-expr'>"
            "<span class='sqrt-symbol'>√</span>"
            f"<span class='radicand'>{escape(match.group(1))}</span>"
            "</span>"
        ),
        text,
    )
    return text


def math_text(value: sp.Expr | float | int) -> str:
    text = sp.sstr(sp.nsimplify(value))
    fraction_match = re.fullmatch(r"(-?[^/]+)/([^/]+)", text)
    if fraction_match:
        numerator, denominator = fraction_match.groups()
        return (
            "<span class='frac'>"
            f"<span class='num'>{pretty_html(numerator)}</span>"
            f"<span class='den'>{pretty_html(denominator)}</span>"
            "</span>"
        )
    return pretty_html(text)


@lru_cache(maxsize=256)
def math_svg(latex: str, width: int | None = None) -> str:
    fig = plt.figure(figsize=(0.01, 0.01), dpi=220)
    fig.patch.set_alpha(0)
    text = fig.text(
        0,
        0,
        f"${latex}$",
        fontsize=15,
        color="#202020",
        fontfamily="serif",
        math_fontfamily="cm",
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    style = f"width:{width}px;" if width else ""
    return f"<img class='math-svg' style='{style}' src='data:image/svg+xml;base64,{encoded}' alt='{escape(latex)}'>"


def expr_svg(value: sp.Expr | float | int, width: int | None = None) -> str:
    return math_svg(sp.latex(sp.nsimplify(value)), width=width)


def merge_points(*point_groups: list[sp.Expr]) -> list[sp.Expr]:
    points: dict[float, sp.Expr] = {}
    for group in point_groups:
        for point in group:
            try:
                points[round(expr_to_float(point), 8)] = sp.simplify(point)
            except Exception:
                continue
    return [points[key] for key in sorted(points)]


def real_roots_in_range(
    expr: sp.Expr,
    x_min: float = -math.inf,
    x_max: float = math.inf,
    limit: int = 12,
) -> list[sp.Expr]:
    roots: dict[float, sp.Expr] = {}
    try:
        for root in sp.solve(sp.together(expr), x):
            if root.is_real or abs(expr_to_float(sp.im(root.evalf()))) < 1e-9:
                value = expr_to_float(root)
                if x_min <= value <= x_max and math.isfinite(value):
                    roots[round(value, 8)] = sp.simplify(root)
    except Exception:
        pass

    if len(roots) < limit:
        try:
            numerator = sp.together(expr).as_numer_denom()[0]
            for root in sp.nroots(numerator, n=30, maxsteps=100):
                if abs(expr_to_float(sp.im(root))) < 1e-8:
                    value = expr_to_float(root)
                    if x_min <= value <= x_max and math.isfinite(value):
                        roots.setdefault(round(value, 8), sp.nsimplify(value))
        except Exception:
            pass

    return [roots[key] for key in sorted(roots)[:limit]]


def vertical_asymptotes(expr: sp.Expr, x_min: float = -math.inf, x_max: float = math.inf) -> list[sp.Expr]:
    asymptotes: dict[float, sp.Expr] = {}
    try:
        denominator = sp.together(expr).as_numer_denom()[1]
        for root in sp.solve(denominator, x):
            if root.is_real or abs(expr_to_float(sp.im(root.evalf()))) < 1e-9:
                value = expr_to_float(root)
                if x_min <= value <= x_max and math.isfinite(value):
                    asymptotes[round(value, 8)] = sp.simplify(root)
    except Exception:
        pass
    return [asymptotes[key] for key in sorted(asymptotes)]


def finite_eval(function, values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        result = np.array(function(values), dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def analyze(raw_expression: str) -> AnalysisResult:
    expr = parse_function(raw_expression)
    first_derivative = sp.simplify(sp.diff(expr, x))
    second_derivative = sp.simplify(sp.diff(first_derivative, x))

    x_intercepts = real_roots_in_range(expr)
    critical_points = real_roots_in_range(first_derivative)
    inflection_candidates = real_roots_in_range(second_derivative)
    asymptotes = vertical_asymptotes(expr)

    y_intercept: float | None
    try:
        y_value = float(expr.subs(x, 0).evalf())
        y_intercept = y_value if math.isfinite(y_value) else None
    except Exception:
        y_intercept = None

    return AnalysisResult(
        expr=expr,
        first_derivative=first_derivative,
        second_derivative=second_derivative,
        x_intercepts=x_intercepts,
        y_intercept=y_intercept,
        critical_points=critical_points,
        inflection_candidates=inflection_candidates,
        vertical_asymptotes=asymptotes,
    )


def sign_label(value: float) -> str:
    if not math.isfinite(value):
        return "판정 어려움"
    if value > 1e-7:
        return "+"
    if value < -1e-7:
        return "-"
    return "0"


def interval_table(expr: sp.Expr, split_points: list[sp.Expr], x_min: float, x_max: float, label: str) -> pd.DataFrame:
    numeric_points = [expr_to_float(point) for point in split_points]
    points = [x_min] + [point for point in numeric_points if x_min < point < x_max] + [x_max]
    function = sp.lambdify(x, expr, "numpy")
    rows: list[dict[str, str]] = []

    for start, end in zip(points[:-1], points[1:]):
        if abs(end - start) < 1e-8:
            continue
        sample = (start + end) / 2
        try:
            value = float(function(sample))
        except Exception:
            value = float("nan")
        rows.append(
            {
                "구간": f"({start:g}, {end:g})",
                "대표값": f"{sample:g}",
                label: sign_label(value),
                "그래프 해석": interpretation(label, sign_label(value)),
            }
        )
    return pd.DataFrame(rows)


def interpretation(label: str, sign: str) -> str:
    if label == "f'(x) 부호":
        return {"+" : "증가", "-" : "감소", "0": "변화 없음"}.get(sign, "추가 확인")
    return {"+" : "아래로 볼록", "-" : "위로 볼록", "0": "변곡 후보"}.get(sign, "추가 확인")


def eval_sign(expr: sp.Expr, value: float | sp.Expr) -> str:
    try:
        return sign_label(expr_to_float(expr.subs(x, value)))
    except Exception:
        return "판정 어려움"


def interval_sign(expr: sp.Expr, start: float, end: float) -> str:
    if not math.isfinite(start) and not math.isfinite(end):
        sample = 0
    elif not math.isfinite(start):
        sample = end - 1
    elif not math.isfinite(end):
        sample = start + 1
    else:
        sample = (start + end) / 2
    return eval_sign(expr, sample)


def curve_arrow(first_sign: str, second_sign: str) -> str:
    paths = {
        ("+", "+"): "M7 43 C17 42 31 31 49 7",
        ("+", "-"): "M7 43 C18 15 34 7 49 7",
        ("-", "+"): "M7 7 C18 34 34 43 49 43",
        ("-", "-"): "M7 7 C20 7 38 18 49 43",
        ("0", "+"): "M7 25 C20 24 34 18 49 7",
        ("0", "-"): "M7 25 C20 26 34 32 49 43",
    }
    path = paths.get((first_sign, second_sign), "M7 25 L49 25")
    marker_key = {
        "+": "plus",
        "-": "minus",
        "0": "zero",
    }
    marker_id = f"arrowhead-{marker_key.get(first_sign, 'unknown')}-{marker_key.get(second_sign, 'unknown')}"
    return f"""
    <svg class="curve-arrow" viewBox="0 0 56 50" aria-hidden="true">
      <defs>
        <marker id="{marker_id}" markerWidth="7" markerHeight="7" refX="5.6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="#404040"></path>
        </marker>
      </defs>
      <path d="{path}" fill="none" stroke="#404040" stroke-width="2.2" stroke-linecap="round"
            marker-end="url(#{marker_id})"></path>
    </svg>
    """


def point_kind(result: AnalysisResult, point: sp.Expr) -> str:
    point_value = round(expr_to_float(point), 8)
    critical_values = {round(expr_to_float(value), 8) for value in result.critical_points}
    inflection_values = {round(expr_to_float(value), 8) for value in result.inflection_candidates}
    asymptote_values = {round(expr_to_float(value), 8) for value in result.vertical_asymptotes}

    labels: list[str] = []
    if point_value in asymptote_values:
        labels.append("점근선")
    if point_value in critical_values:
        left_sign = eval_sign(result.first_derivative, point_value - 1e-4)
        right_sign = eval_sign(result.first_derivative, point_value + 1e-4)
        if left_sign == "+" and right_sign == "-":
            labels.append("극대")
        elif left_sign == "-" and right_sign == "+":
            labels.append("극소")
        else:
            labels.append("극점")
    if point_value in inflection_values:
        labels.append("변곡점")
    return ", ".join(labels)


def function_value_cell(result: AnalysisResult, point: sp.Expr) -> str:
    kind = point_kind(result, point)
    if "점근선" in kind:
        return "<span class='chart-note'>(점근선)</span>"
    try:
        value = sp.simplify(result.expr.subs(x, point))
        if kind:
            return f"{expr_svg(value)}<br><span class='chart-note'>({escape(kind)})</span>"
        return expr_svg(value)
    except Exception:
        return f"<span class='chart-note'>({escape(kind or '계산 필요')})</span>"


def sign_chart_html(result: AnalysisResult) -> str:
    split_points = merge_points(
        result.critical_points,
        result.inflection_candidates,
        result.vertical_asymptotes,
    )
    numeric_points = [expr_to_float(point) for point in split_points]
    boundaries = [-math.inf] + numeric_points + [math.inf]

    x_head = math_svg("x", width=18)
    first_head = math_svg("f'(x)", width=54)
    second_head = math_svg("f''(x)", width=58)
    function_head = math_svg("f(x)", width=48)
    x_cells = [f"<th class='row-head'>{x_head}</th>"]
    first_cells = [f"<th class='row-head'>{first_head}</th>"]
    second_cells = [f"<th class='row-head'>{second_head}</th>"]
    function_cells = [f"<th class='row-head'>{function_head}</th>"]

    for index, point in enumerate(split_points):
        start, end = boundaries[index], boundaries[index + 1]
        first_interval = interval_sign(result.first_derivative, start, end)
        second_interval = interval_sign(result.second_derivative, start, end)

        x_cells.append("<td class='interval'>⋯</td>")
        first_cells.append(f"<td class='interval'>{escape(first_interval)}</td>")
        second_cells.append(f"<td class='interval'>{escape(second_interval)}</td>")
        function_cells.append(f"<td class='interval arrow'>{curve_arrow(first_interval, second_interval)}</td>")

        x_cells.append(f"<td class='point'>{expr_svg(point)}</td>")
        first_cells.append(f"<td class='point'>{escape(eval_sign(result.first_derivative, point))}</td>")
        second_cells.append(f"<td class='point'>{escape(eval_sign(result.second_derivative, point))}</td>")
        function_cells.append(f"<td class='point fx'>{function_value_cell(result, point)}</td>")

    final_start, final_end = boundaries[-2], boundaries[-1]
    first_interval = interval_sign(result.first_derivative, final_start, final_end)
    second_interval = interval_sign(result.second_derivative, final_start, final_end)
    x_cells.append("<td class='interval'>⋯</td>")
    first_cells.append(f"<td class='interval'>{escape(first_interval)}</td>")
    second_cells.append(f"<td class='interval'>{escape(second_interval)}</td>")
    function_cells.append(f"<td class='interval arrow'>{curve_arrow(first_interval, second_interval)}</td>")

    return f"""
<style>
.sign-chart-wrap {{
    overflow-x: auto;
    padding: 0.25rem 0 0.5rem;
}}
.sign-chart {{
    border-collapse: separate;
    border-spacing: 0;
    min-width: 780px;
    width: 100%;
    table-layout: fixed;
    border: 2px solid #c7c7c7;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
    font-size: 1.08rem;
    text-align: center;
    font-family: "Noto Sans KR", "Malgun Gothic", Arial, sans-serif;
}}
.sign-chart th,
.sign-chart td {{
    border-right: 1.6px solid #d0d0d0;
    border-bottom: 1.6px solid #d0d0d0;
    min-width: 64px;
    height: 46px;
    padding: 0.22rem 0.35rem;
    vertical-align: middle;
}}
.sign-chart tr:last-child th,
.sign-chart tr:last-child td {{
    border-bottom: 0;
}}
.sign-chart th:last-child,
.sign-chart td:last-child {{
    border-right: 0;
}}
.sign-chart .row-head {{
    width: 84px;
    min-width: 84px;
    background: #cfe8b1;
    color: #26351e;
    font-weight: 400;
    font-style: normal;
    border-right: 2px solid #c7c7c7;
}}
.sign-chart .interval {{
    color: #525252;
    background: #fbfbfb;
}}
.sign-chart .point {{
    color: #252525;
    background: #ffffff;
    font-weight: 600;
}}
.sign-chart .arrow {{
    padding: 0.1rem 0.2rem;
}}
.sign-chart .curve-arrow {{
    width: 54px;
    height: 44px;
    display: inline-block;
    vertical-align: middle;
}}
.sign-chart .fx {{
    line-height: 1.2;
    min-height: 68px;
}}
.sign-chart .chart-note {{
    color: #333;
    font-size: 0.9rem;
    font-weight: 500;
}}
.sign-chart .math-svg {{
    display: inline-block;
    max-width: 100%;
    height: auto;
    vertical-align: middle;
}}
</style>
<div class="sign-chart-wrap">
  <table class="sign-chart">
    <tbody>
      <tr>{''.join(x_cells)}</tr>
      <tr>{''.join(first_cells)}</tr>
      <tr>{''.join(second_cells)}</tr>
      <tr>{''.join(function_cells)}</tr>
    </tbody>
  </table>
</div>
"""


def point_table(result: AnalysisResult) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for value in result.x_intercepts:
        rows.append({"종류": "x절편", "x": exact_text(value), "y": "0"})
    if result.y_intercept is not None:
        rows.append({"종류": "y절편", "x": "0", "y": exact_text(result.y_intercept)})
    for value in result.critical_points:
        try:
            y_value = sp.simplify(result.expr.subs(x, value))
            rows.append({"종류": point_kind(result, value) or "극점", "x": exact_text(value), "y": exact_text(y_value)})
        except Exception:
            rows.append({"종류": "극점", "x": exact_text(value), "y": "계산 필요"})
    for value in result.inflection_candidates:
        try:
            y_value = sp.simplify(result.expr.subs(x, value))
            rows.append({"종류": "변곡점", "x": exact_text(value), "y": exact_text(y_value)})
        except Exception:
            rows.append({"종류": "변곡점", "x": exact_text(value), "y": "계산 필요"})
    for value in result.vertical_asymptotes:
        rows.append({"종류": "수직점근선 후보", "x": exact_text(value), "y": "정의되지 않음"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["종류", "x", "y"])


def auto_x_window(result: AnalysisResult) -> tuple[float, float]:
    points = [
        expr_to_float(point)
        for point in merge_points(
            result.x_intercepts,
            result.critical_points,
            result.inflection_candidates,
            result.vertical_asymptotes,
        )
    ]
    if not points:
        return -6.0, 6.0

    left = min(points)
    right = max(points)
    span = max(right - left, 4.0)
    padding = max(2.0, span * 0.35)
    return left - padding, right + padding


def graph_sample_window(initial_window: tuple[float, float], result: AnalysisResult) -> tuple[float, float]:
    left, right = initial_window
    center = (left + right) / 2
    visible_width = max(right - left, 12.0)
    sample_width = max(1000.0, visible_width * 30)

    important_points = [
        expr_to_float(point)
        for point in merge_points(
            result.x_intercepts,
            result.critical_points,
            result.inflection_candidates,
            result.vertical_asymptotes,
        )
    ]
    if important_points:
        point_left = min(important_points)
        point_right = max(important_points)
        sample_left = min(center - sample_width / 2, point_left - visible_width * 2)
        sample_right = max(center + sample_width / 2, point_right + visible_width * 2)
        return sample_left, sample_right

    return center - sample_width / 2, center + sample_width / 2


def auto_y_window(ys: np.ndarray) -> tuple[float, float] | None:
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        return None

    low, high = np.nanpercentile(finite, [2, 98])
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    if abs(high - low) < 1e-8:
        padding = max(1.0, abs(high) * 0.3)
        return low - padding, high + padding

    padding = (high - low) * 0.18
    return low - padding, high + padding


def add_axis_coordinate_guides(fig: go.Figure, points: list[tuple[sp.Expr, sp.Expr]], color: str) -> None:
    for x_value, y_value in points:
        numeric_x = expr_to_float(x_value)
        numeric_y = expr_to_float(y_value)
        x_label = exact_text(x_value)
        y_label = exact_text(y_value)
        fig.add_shape(
            type="line",
            x0=numeric_x,
            x1=numeric_x,
            y0=0,
            y1=numeric_y,
            line={"color": color, "width": 1.5, "dash": "dot"},
        )
        fig.add_shape(
            type="line",
            x0=0,
            x1=numeric_x,
            y0=numeric_y,
            y1=numeric_y,
            line={"color": color, "width": 1.5, "dash": "dot"},
        )
        fig.add_annotation(
            x=numeric_x,
            y=0,
            text=f"x={x_label}",
            showarrow=False,
            yshift=-18,
            font={"size": 12, "color": color},
            bgcolor="rgba(255,255,255,0.82)",
        )
        fig.add_annotation(
            x=0,
            y=numeric_y,
            text=f"y={y_label}",
            showarrow=False,
            xshift=36,
            font={"size": 12, "color": color},
            bgcolor="rgba(255,255,255,0.82)",
        )


def make_graph(result: AnalysisResult, show_extrema_coordinates: bool, show_inflection_coordinates: bool) -> go.Figure:
    initial_x_min, initial_x_max = auto_x_window(result)
    x_min, x_max = graph_sample_window((initial_x_min, initial_x_max), result)
    expression_function = sp.lambdify(x, result.expr, "numpy")
    xs = np.linspace(x_min, x_max, 10000)

    for point in result.vertical_asymptotes:
        numeric_point = expr_to_float(point)
        xs = xs[np.abs(xs - numeric_point) > (x_max - x_min) / 1600]

    ys = finite_eval(expression_function, xs)
    visible_mask = (xs >= initial_x_min) & (xs <= initial_x_max)
    y_window = auto_y_window(ys[visible_mask])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="f(x)",
            line={"color": "#2563eb", "width": 3},
        )
    )

    marker_colors = {
        "x절편": "#0f766e",
        "y절편": "#0f766e",
        "극점": "#dc2626",
        "변곡점": "#7c3aed",
    }
    graph_points: list[dict[str, float | str]] = []
    for value in result.x_intercepts:
        graph_points.append(
            {"종류": "x절편", "x": expr_to_float(value), "y": 0.0, "label": f"x절편<br>x={exact_text(value)}"}
        )
    if result.y_intercept is not None:
        graph_points.append({"종류": "y절편", "x": 0.0, "y": result.y_intercept, "label": "y절편<br>x=0"})
    for kind, values in {
        "극점": result.critical_points,
        "변곡점": result.inflection_candidates,
    }.items():
        for value in values:
            try:
                graph_points.append(
                    {
                        "종류": kind,
                        "x": expr_to_float(value),
                        "y": expr_to_float(result.expr.subs(x, value)),
                        "label": f"{kind}<br>x={exact_text(value)}",
                    }
                )
            except Exception:
                continue

    points = pd.DataFrame(graph_points, columns=["종류", "x", "y", "label"])
    if not points.empty:
        for kind, group in points.groupby("종류"):
            fig.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["y"],
                    mode="markers",
                    hovertext=group.get("label", pd.Series([kind] * len(group))),
                    hovertemplate="%{hovertext}<br>y=%{y:.6g}<extra></extra>",
                    name=kind,
                    marker={"size": 10, "color": marker_colors[kind]},
                )
            )

    for point in result.vertical_asymptotes:
        numeric_point = expr_to_float(point)
        fig.add_vline(
            x=numeric_point,
            line_dash="dash",
            line_color="#ea580c",
            annotation_text=f"x={exact_text(point)}",
            annotation_position="top",
        )

    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)

    if show_extrema_coordinates:
        extrema_points = []
        for point in result.critical_points:
            try:
                extrema_points.append((point, sp.simplify(result.expr.subs(x, point))))
            except Exception:
                continue
        add_axis_coordinate_guides(fig, extrema_points, "#dc2626")

    if show_inflection_coordinates:
        inflection_points = []
        for point in result.inflection_candidates:
            try:
                inflection_points.append((point, sp.simplify(result.expr.subs(x, point))))
            except Exception:
                continue
        add_axis_coordinate_guides(fig, inflection_points, "#7c3aed")

    fig.update_layout(
        height=560,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    fig.update_xaxes(range=[initial_x_min, initial_x_max])
    if y_window is not None:
        fig.update_yaxes(range=list(y_window))
    return fig


def report_text(result: AnalysisResult, notes: str) -> str:
    points = point_table(result).to_markdown(index=False)
    return f"""# 함수 그래프 개형 검증 보고서

함수: `{sp.sstr(result.expr)}`

1차 도함수: `{sp.sstr(result.first_derivative)}`

2차 도함수: `{sp.sstr(result.second_derivative)}`

## 주요 지점

{points}

## 비교 및 성찰

{notes.strip() or "아직 작성하지 않았습니다."}
"""


def main() -> None:
    st.set_page_config(
        page_title="함수 그래프 개형 검증 도구",
        page_icon="📈",
        layout="wide",
    )

    st.title("함수의 증감표를 이용한 그래프 개형 검증")
    st.caption("손으로 작성한 증감표와 그래프가 수학적으로 맞는지 확인하는 Streamlit 수업 도구")

    with st.sidebar:
        st.header("함수 입력")
        input_mode = st.radio("입력 형식", ["일반식", "분수식"], horizontal=True)
        if input_mode == "일반식":
            raw_expression = st.text_input("f(x)", value="", placeholder="예: sin(x), ln(x), x/(x^2+1)")
        else:
            numerator = st.text_input("분자", value="", placeholder="예: x")
            denominator = st.text_input("분모", value="", placeholder="예: x^2+1")
            raw_expression = f"({numerator})/({denominator})" if numerator.strip() and denominator.strip() else ""
            if raw_expression:
                st.caption(f"적용식: `{raw_expression}`")
        st.info("거듭제곱은 `^` 또는 `**`를 사용할 수 있습니다. 예: `x/(x^2+1)`")
        st.divider()
        show_extrema_coordinates = st.toggle("극점 좌표를 축에 표시", value=False)
        show_inflection_coordinates = st.toggle("변곡점 좌표를 축에 표시", value=False)

    if not raw_expression.strip():
        st.markdown(
            """
            <div style="border:1px solid #d9e2ec;border-radius:10px;padding:2rem;background:#ffffff;">
              <h3 style="margin-top:0;">함수식을 입력하면 그래프와 증감표가 생성됩니다.</h3>
              <p style="margin-bottom:0;color:#52616b;">
                왼쪽 입력창에 일반식 또는 분수식을 넣어주세요. 예: <code>sin(x)</code>, <code>ln(x)</code>, <code>x/(x^2+1)</code>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        result = analyze(raw_expression)
    except Exception as exc:
        st.error(f"함수식을 해석하지 못했습니다: {exc}")
        st.stop()

    graph_tab, table_tab = st.tabs(["그래프 확인", "증감표 분석"])

    with graph_tab:
        left, right = st.columns([2.2, 1])
        with left:
            st.plotly_chart(
                make_graph(result, show_extrema_coordinates, show_inflection_coordinates),
                width="stretch",
            )
        with right:
            st.subheader("미분 결과")
            st.latex(f"f(x) = {sp.latex(result.expr)}")
            st.latex(f"f'(x) = {sp.latex(result.first_derivative)}")
            st.latex(f"f''(x) = {sp.latex(result.second_derivative)}")

    with table_tab:
        st.subheader("증감표")
        st.markdown(sign_chart_html(result), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
