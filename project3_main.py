# -*- coding: utf-8 -*-
"""
Project 3 - Numerical Methods
Sections:
1) Lagrange interpolation and natural cubic spline
2) Read Excel data and draw bar, line, and box plots
3) Numerical integration of exp(x^2) on [0, 1]

Run:
    python project3_main.py

Important:
    Put this file beside the Excel file named: project3.xlsx
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "project3.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
RESULTS_FILE = OUTPUT_DIR / "results.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def write_result(text: str, mode: str = "a") -> None:
    """Print output and also save it in outputs/results.txt."""
    print(text)
    with open(RESULTS_FILE, mode, encoding="utf-8") as f:
        f.write(text + "\n")


def extract_number(value) -> float:
    """Extract numeric part from strings like '100KB'."""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"Cannot extract numeric data size from: {value}")
    return float(match.group())


# =========================
# Section 1: Interpolation
# =========================
def lagrange_interpolation(points):
    x = sp.symbols("x")
    polynomial = sp.interpolate(points, x)
    return sp.expand(polynomial)


def natural_cubic_spline(points):
    """
    Build exact natural cubic spline formulas using SymPy.
    Natural spline condition: second derivative at first and last point is zero.
    """
    x = sp.symbols("x")
    xs = [sp.Rational(p[0]) for p in points]
    ys = [sp.Rational(p[1]) for p in points]
    n = len(points) - 1

    variables = []
    coeffs = []

    # S_i(x) = a_i*x^3 + b_i*x^2 + c_i*x + d_i
    for i in range(n):
        a, b, c, d = sp.symbols(f"a{i} b{i} c{i} d{i}")
        coeffs.append((a, b, c, d))
        variables.extend([a, b, c, d])

    equations = []

    # Each spline piece must pass through the two endpoint data points
    for i in range(n):
        a, b, c, d = coeffs[i]
        S = a * x**3 + b * x**2 + c * x + d
        equations.append(sp.Eq(S.subs(x, xs[i]), ys[i]))
        equations.append(sp.Eq(S.subs(x, xs[i + 1]), ys[i + 1]))

    # First and second derivatives must be continuous at internal points
    for i in range(1, n):
        xi = xs[i]
        a0, b0, c0, d0 = coeffs[i - 1]
        a1, b1, c1, d1 = coeffs[i]
        S0 = a0 * x**3 + b0 * x**2 + c0 * x + d0
        S1 = a1 * x**3 + b1 * x**2 + c1 * x + d1
        equations.append(sp.Eq(sp.diff(S0, x).subs(x, xi), sp.diff(S1, x).subs(x, xi)))
        equations.append(sp.Eq(sp.diff(S0, x, 2).subs(x, xi), sp.diff(S1, x, 2).subs(x, xi)))

    # Natural boundary conditions
    a_first, b_first, c_first, d_first = coeffs[0]
    S_first = a_first * x**3 + b_first * x**2 + c_first * x + d_first
    equations.append(sp.Eq(sp.diff(S_first, x, 2).subs(x, xs[0]), 0))

    a_last, b_last, c_last, d_last = coeffs[-1]
    S_last = a_last * x**3 + b_last * x**2 + c_last * x + d_last
    equations.append(sp.Eq(sp.diff(S_last, x, 2).subs(x, xs[-1]), 0))

    solution = sp.solve(equations, variables, dict=True)[0]

    spline_pieces = []
    for i in range(n):
        a, b, c, d = coeffs[i]
        S = sp.expand((a * x**3 + b * x**2 + c * x + d).subs(solution))
        spline_pieces.append((xs[i], xs[i + 1], S))

    return spline_pieces


def evaluate_spline(spline_pieces, x_values):
    x = sp.symbols("x")
    y_values = []
    for value in x_values:
        selected = None
        for start, end, expr in spline_pieces:
            if float(start) <= value <= float(end):
                selected = expr
                break
        if selected is None:
            selected = spline_pieces[-1][2]
        y_values.append(float(selected.subs(x, value)))
    return np.array(y_values)


def run_interpolation_section():
    write_result("\n========== Section 1: Interpolation ==========")

    x = sp.symbols("x")
    points = [(1, 1), (2, 3), (3, 5), (4, 8), (5, 5), (6, 2)]

    lagrange_poly = lagrange_interpolation(points)
    write_result("Lagrange polynomial:")
    write_result(f"P(x) = {lagrange_poly}")

    spline_pieces = natural_cubic_spline(points)
    write_result("\nNatural cubic spline pieces:")
    for start, end, expr in spline_pieces:
        write_result(f"For {start} <= x <= {end}:  S(x) = {expr}")

    # Optional plot for report
    x_plot = np.linspace(1, 6, 300)
    lagrange_func = sp.lambdify(x, lagrange_poly, "numpy")
    y_lagrange = lagrange_func(x_plot)
    y_spline = evaluate_spline(spline_pieces, x_plot)

    px = [p[0] for p in points]
    py = [p[1] for p in points]

    plt.figure(figsize=(9, 5))
    plt.scatter(px, py, label="Data points", zorder=5)
    plt.plot(x_plot, y_lagrange, label="Lagrange interpolation")
    plt.plot(x_plot, y_spline, label="Natural cubic spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lagrange vs Natural Cubic Spline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = CHART_DIR / "interpolation_lagrange_spline.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    write_result(f"Interpolation chart saved: {output_path}")


# =========================
# Section 2: Excel + Charts
# =========================
def read_excel_data():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Excel file not found: {EXCEL_FILE}\n"
            "Put the Excel file beside project3_main.py or edit EXCEL_FILE path."
        )

    df = pd.read_excel(EXCEL_FILE)
    df.columns = [str(col).strip() for col in df.columns]

    if len(df.columns) < 4:
        raise ValueError("Excel file must have at least 4 columns: Data size, Alg.1, Alg.2, Alg.3")

    data_size_col = df.columns[0]
    algorithm_cols = list(df.columns[1:])

    # Convert algorithm columns to numeric values
    for col in algorithm_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make a numeric helper column for correct sorting and filtering
    df["DataSizeNumber"] = df[data_size_col].apply(extract_number)
    df = df.sort_values("DataSizeNumber").reset_index(drop=True)

    return df, data_size_col, algorithm_cols


def plot_bar_chart(df, data_size_col, algorithm_cols):
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(df))
    width = 0.8 / len(algorithm_cols)

    for i, col in enumerate(algorithm_cols):
        plt.bar(x_positions + i * width, df[col], width=width, label=col)

    center_positions = x_positions + width * (len(algorithm_cols) - 1) / 2
    plt.xticks(center_positions, df[data_size_col].astype(str))
    plt.xlabel("Data size")
    plt.ylabel("Execution time")
    plt.title("Execution Time of Algorithms - Bar Chart")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = CHART_DIR / "bar_chart.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_line_chart(df, data_size_col, algorithm_cols):
    plt.figure(figsize=(10, 6))

    for col in algorithm_cols:
        plt.plot(df[data_size_col].astype(str), df[col], marker="o", linewidth=2, label=col)

    plt.xlabel("Data size")
    plt.ylabel("Execution time")
    plt.title("Execution Time of Algorithms - Line Chart")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = CHART_DIR / "line_chart.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_box_chart(df, algorithm_cols):
    plt.figure(figsize=(8, 6))
    plt.boxplot([df[col].dropna() for col in algorithm_cols], tick_labels=algorithm_cols)
    plt.xlabel("Algorithms")
    plt.ylabel("Execution time")
    plt.title("Execution Time Distribution - Box Plot")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = CHART_DIR / "box_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def run_excel_chart_section():
    write_result("\n========== Section 2: Excel Reading and Charts ==========")

    df, data_size_col, algorithm_cols = read_excel_data()

    write_result("Data read from Excel:")
    write_result(df[[data_size_col] + algorithm_cols].to_string(index=False))

    bar_path = plot_bar_chart(df, data_size_col, algorithm_cols)
    line_path = plot_line_chart(df, data_size_col, algorithm_cols)
    box_path = plot_box_chart(df, algorithm_cols)

    write_result(f"Bar chart saved: {bar_path}")
    write_result(f"Line chart saved: {line_path}")
    write_result(f"Box plot saved: {box_path}")

    # Requirement: mean of Alg.2 for 100KB to 600KB only
    alg2_col = None
    for col in algorithm_cols:
        if col.replace(" ", "").lower() in ["alg.2", "alg2", "algorithm2"]:
            alg2_col = col
            break

    if alg2_col is None:
        # fallback: second algorithm column
        alg2_col = algorithm_cols[1]

    filtered_df = df[(df["DataSizeNumber"] >= 100) & (df["DataSizeNumber"] <= 600)]
    alg2_mean = filtered_df[alg2_col].mean()

    write_result(f"Mean execution time of {alg2_col} for 100KB to 600KB = {alg2_mean}")

    write_result("\nChart analysis:")
    write_result("Alg.1 has the lowest execution time and grows slowly as data size increases.")
    write_result("Alg.2 has a medium execution time and shows a steady increasing trend.")
    write_result("Alg.3 has the highest growth rate and becomes much slower for larger input sizes.")


# =========================
# Section 3: Numerical Integration
# =========================
def f(x):
    return np.exp(x**2)


def trapezoidal_rule(func, a, b, n):
    x_values = np.linspace(a, b, n + 1)
    y_values = func(x_values)
    h = (b - a) / n
    result = h * (0.5 * y_values[0] + np.sum(y_values[1:-1]) + 0.5 * y_values[-1])
    return result


def simpson_rule(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")

    x_values = np.linspace(a, b, n + 1)
    y_values = func(x_values)
    h = (b - a) / n
    result = (h / 3) * (
        y_values[0]
        + y_values[-1]
        + 4 * np.sum(y_values[1:-1:2])
        + 2 * np.sum(y_values[2:-1:2])
    )
    return result


def gaussian_quadrature(func, a, b, number_of_points):
    """Gauss-Legendre quadrature on [a, b]."""
    nodes, weights = np.polynomial.legendre.leggauss(number_of_points)

    # Transform from [-1, 1] to [a, b]
    transformed_nodes = ((b - a) / 2) * nodes + ((a + b) / 2)
    transformed_weights = ((b - a) / 2) * weights

    return np.sum(transformed_weights * func(transformed_nodes))


def run_integration_section():
    write_result("\n========== Section 3: Numerical Integration ==========")

    a, b = 0, 1
    n = 1000

    trapezoidal_result = trapezoidal_rule(f, a, b, n)
    simpson_result = simpson_rule(f, a, b, n)
    gaussian_result_5 = gaussian_quadrature(f, a, b, 5)
    gaussian_result_10 = gaussian_quadrature(f, a, b, 10)

    # A high-point Gaussian result is used as a very accurate reference for comparison
    reference_result = gaussian_quadrature(f, a, b, 50)

    write_result("Integral of f(x)=exp(x^2) on [0,1]")
    write_result(f"Trapezoidal rule, n={n}: {trapezoidal_result:.12f}")
    write_result(f"Simpson rule, n={n}:     {simpson_result:.12f}")
    write_result(f"Gaussian quadrature, 5 points:  {gaussian_result_5:.12f}")
    write_result(f"Gaussian quadrature, 10 points: {gaussian_result_10:.12f}")
    write_result(f"Reference value: {reference_result:.12f}")

    write_result("\nAbsolute errors compared with reference value:")
    write_result(f"Trapezoidal error: {abs(trapezoidal_result - reference_result):.12e}")
    write_result(f"Simpson error:     {abs(simpson_result - reference_result):.12e}")
    write_result(f"Gaussian 5 error:  {abs(gaussian_result_5 - reference_result):.12e}")
    write_result(f"Gaussian 10 error: {abs(gaussian_result_10 - reference_result):.12e}")



# =========================
# Main
# =========================
def main():
    # Clear previous results file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f_out:
        f_out.write("Project 3 Results\n")
        f_out.write("=================\n")

    run_interpolation_section()
    run_excel_chart_section()
    run_integration_section()

    write_result("\nAll outputs were saved in the outputs folder.")


if __name__ == "__main__":
    main()
