from __future__ import annotations

import math
from dataclasses import dataclass

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
    x_intercepts: list[float]
    y_intercept: float | None
    critical_points: list[float]
    inflection_candidates: list[float]
    vertical_asymptotes: list[float]


def parse_function(raw_expression: str) -> sp.Expr:
    cleaned = raw_expression.strip()
    if cleaned.startswith("y="):
        cleaned = cleaned[2:].strip()
    if cleaned.startswith("f(x)="):
        cleaned = cleaned[5:].strip()
    return sp.simplify(
        parse_expr(
            cleaned,
            local_dict={"x": x, "e": sp.E, "pi": sp.pi},
            transformations=TRANSFORMATIONS,
        )
    )


def real_roots_in_range(expr: sp.Expr, x_min: float, x_max: float, limit: int = 12) -> list[float]:
    roots: set[float] = set()
    try:
        for root in sp.solve(sp.together(expr), x):
            if root.is_real or abs(float(sp.im(root.evalf()))) < 1e-9:
                value = float(sp.re(root.evalf()))
                if x_min <= value <= x_max and math.isfinite(value):
                    roots.add(round(value, 8))
    except Exception:
        pass

    if len(roots) < limit:
        try:
            numerator = sp.together(expr).as_numer_denom()[0]
            for root in sp.nroots(numerator, n=30, maxsteps=100):
                if abs(float(sp.im(root))) < 1e-8:
                    value = float(sp.re(root))
                    if x_min <= value <= x_max and math.isfinite(value):
                        roots.add(round(value, 8))
        except Exception:
            pass

    return sorted(roots)[:limit]


def vertical_asymptotes(expr: sp.Expr, x_min: float, x_max: float) -> list[float]:
    asymptotes: set[float] = set()
    try:
        denominator = sp.together(expr).as_numer_denom()[1]
        for root in sp.solve(denominator, x):
            if root.is_real or abs(float(sp.im(root.evalf()))) < 1e-9:
                value = float(sp.re(root.evalf()))
                if x_min <= value <= x_max and math.isfinite(value):
                    asymptotes.add(round(value, 8))
    except Exception:
        pass
    return sorted(asymptotes)


def finite_eval(function, values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        result = np.array(function(values), dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def analyze(raw_expression: str, x_min: float, x_max: float) -> AnalysisResult:
    expr = parse_function(raw_expression)
    first_derivative = sp.simplify(sp.diff(expr, x))
    second_derivative = sp.simplify(sp.diff(first_derivative, x))

    x_intercepts = real_roots_in_range(expr, x_min, x_max)
    critical_points = real_roots_in_range(first_derivative, x_min, x_max)
    inflection_candidates = real_roots_in_range(second_derivative, x_min, x_max)
    asymptotes = vertical_asymptotes(expr, x_min, x_max)

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


def interval_table(expr: sp.Expr, split_points: list[float], x_min: float, x_max: float, label: str) -> pd.DataFrame:
    points = [x_min] + [point for point in split_points if x_min < point < x_max] + [x_max]
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


def point_table(result: AnalysisResult) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for value in result.x_intercepts:
        rows.append({"종류": "x절편", "x": f"{value:g}", "y": "0"})
    if result.y_intercept is not None:
        rows.append({"종류": "y절편", "x": "0", "y": f"{result.y_intercept:g}"})
    for value in result.critical_points:
        try:
            y_value = float(result.expr.subs(x, value).evalf())
            rows.append({"종류": "극값 후보", "x": f"{value:g}", "y": f"{y_value:g}"})
        except Exception:
            rows.append({"종류": "극값 후보", "x": f"{value:g}", "y": "계산 필요"})
    for value in result.inflection_candidates:
        try:
            y_value = float(result.expr.subs(x, value).evalf())
            rows.append({"종류": "변곡점 후보", "x": f"{value:g}", "y": f"{y_value:g}"})
        except Exception:
            rows.append({"종류": "변곡점 후보", "x": f"{value:g}", "y": "계산 필요"})
    for value in result.vertical_asymptotes:
        rows.append({"종류": "수직점근선 후보", "x": f"{value:g}", "y": "정의되지 않음"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["종류", "x", "y"])


def make_graph(result: AnalysisResult, x_min: float, x_max: float, y_clip: float) -> go.Figure:
    expression_function = sp.lambdify(x, result.expr, "numpy")
    xs = np.linspace(x_min, x_max, 1600)

    for point in result.vertical_asymptotes:
        xs = xs[np.abs(xs - point) > (x_max - x_min) / 1600]

    ys = finite_eval(expression_function, xs)
    ys[np.abs(ys) > y_clip] = np.nan

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

    points = point_table(result)
    marker_colors = {
        "x절편": "#0f766e",
        "y절편": "#0f766e",
        "극값 후보": "#dc2626",
        "변곡점 후보": "#7c3aed",
    }
    for kind, group in points[points["종류"].isin(marker_colors)].groupby("종류"):
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(group["x"], errors="coerce"),
                y=pd.to_numeric(group["y"], errors="coerce"),
                mode="markers+text",
                text=[kind] * len(group),
                textposition="top center",
                name=kind,
                marker={"size": 10, "color": marker_colors[kind]},
            )
        )

    for point in result.vertical_asymptotes:
        fig.add_vline(
            x=point,
            line_dash="dash",
            line_color="#ea580c",
            annotation_text=f"x={point:g}",
            annotation_position="top",
        )

    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        height=560,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    fig.update_yaxes(range=[-y_clip, y_clip])
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
        example = st.selectbox(
            "예시 선택",
            [
                "x/(x^2+1)",
                "x^3 - 3*x",
                "x^4 - 4*x^2",
                "sin(x)",
                "exp(-x^2)",
            ],
        )
        raw_expression = st.text_input("f(x)", value=example)
        st.divider()
        x_min, x_max = st.slider("x 범위", -20.0, 20.0, (-6.0, 6.0), 0.5)
        y_clip = st.slider("y 표시 범위", 2.0, 50.0, 8.0, 1.0)
        st.info("거듭제곱은 `^` 또는 `**`를 사용할 수 있습니다. 예: `x/(x^2+1)`")

    try:
        result = analyze(raw_expression, x_min, x_max)
    except Exception as exc:
        st.error(f"함수식을 해석하지 못했습니다: {exc}")
        st.stop()

    graph_tab, table_tab, reflection_tab, rubric_tab = st.tabs(
        ["그래프 확인", "증감표 분석", "비교·성찰", "평가 기준"]
    )

    with graph_tab:
        left, right = st.columns([2.2, 1])
        with left:
            st.plotly_chart(make_graph(result, x_min, x_max, y_clip), width="stretch")
        with right:
            st.subheader("미분 결과")
            st.latex(f"f(x) = {sp.latex(result.expr)}")
            st.latex(f"f'(x) = {sp.latex(result.first_derivative)}")
            st.latex(f"f''(x) = {sp.latex(result.second_derivative)}")
            st.dataframe(point_table(result), hide_index=True, width="stretch")

    with table_tab:
        st.subheader("증가·감소")
        st.dataframe(
            interval_table(
                result.first_derivative,
                sorted(set(result.critical_points + result.vertical_asymptotes)),
                x_min,
                x_max,
                "f'(x) 부호",
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader("오목·볼록")
        st.dataframe(
            interval_table(
                result.second_derivative,
                sorted(set(result.inflection_candidates + result.vertical_asymptotes)),
                x_min,
                x_max,
                "f''(x) 부호",
            ),
            hide_index=True,
            width="stretch",
        )

    with reflection_tab:
        st.subheader("손그래프와 앱 결과 비교")
        notes = st.text_area(
            "차이가 있었다면 원인을 수학적으로 적어보세요.",
            height=180,
            placeholder="예: f'(x)의 부호를 잘못 판정해서 감소 구간을 증가 구간으로 표시했다.",
        )
        st.download_button(
            "검증 보고서 내려받기",
            data=report_text(result, notes),
            file_name="function_graph_report.md",
            mime="text/markdown",
        )

    with rubric_tab:
        st.subheader("수업 평가 기준")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "평가 요소": "수학적 추론 및 수행",
                        "상": "도함수와 이계도함수를 정확히 활용하여 증감, 오목·볼록, 점근선을 반영한다.",
                        "중": "증감표는 대체로 정확하나 일부 그래프 개형 해석이 미흡하다.",
                        "하": "미분 계산 또는 부호 조사와 그래프 연결에 어려움이 있다.",
                    },
                    {
                        "평가 요소": "정보처리 역량",
                        "상": "웹 앱 결과를 극값, 변곡점 등 수학 개념과 연결하여 설명한다.",
                        "중": "그래프는 도출하지만 수학적 해석에는 일부 도움이 필요하다.",
                        "하": "교사의 도움 없이는 도구 활용이 어렵다.",
                    },
                    {
                        "평가 요소": "비판적 사고 및 성찰",
                        "상": "손그래프와 앱 결과의 차이를 근거 있게 분석하고 수정한다.",
                        "중": "오류 지점은 찾지만 수정 과정에 피드백이 필요하다.",
                        "하": "차이를 발견하거나 원인을 설명하는 데 어려움이 있다.",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
