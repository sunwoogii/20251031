from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


###############################################################################
# Data loading utilities
###############################################################################


@dataclass
class ReturnSeries:
    ticker: str
    dates: List[datetime]
    returns: List[float]

    def to_dict(self) -> Dict[datetime, float]:
        return {date: ret for date, ret in zip(self.dates, self.returns)}


def load_close_returns(csv_path: Path) -> List[ReturnSeries]:
    """Load daily close-to-close returns for every ticker in the CSV."""

    with csv_path.open() as f:
        reader = csv.reader(f)
        header1, header2, header3 = [next(reader) for _ in range(3)]
        rows = list(reader)

    close_columns: Dict[str, int] = {}
    for idx, (metric, ticker) in enumerate(zip(header1[1:], header2[1:]), start=1):
        if metric == "Close":
            close_columns[ticker] = idx

    close_prices: Dict[str, List[Tuple[datetime, float]]] = {
        ticker: [] for ticker in close_columns
    }
    for row in rows:
        if not row:
            continue
        date = datetime.strptime(row[0], "%Y-%m-%d")
        for ticker, col_idx in close_columns.items():
            value = row[col_idx]
            if value:
                close_prices[ticker].append((date, float(value)))

    series_list: List[ReturnSeries] = []
    for ticker, price_series in close_prices.items():
        price_series.sort(key=lambda item: item[0])
        dates: List[datetime] = []
        returns: List[float] = []
        for (prev_date, prev_price), (curr_date, curr_price) in zip(
            price_series, price_series[1:]
        ):
            if prev_price != 0:
                dates.append(curr_date)
                returns.append(curr_price / prev_price - 1.0)
        series_list.append(ReturnSeries(ticker=ticker, dates=dates, returns=returns))

    series_list.sort(key=lambda item: item.ticker)
    return series_list


###############################################################################
# Linear algebra helpers
###############################################################################


def align_returns(series_list: Sequence[ReturnSeries]) -> Tuple[List[datetime], List[List[float]]]:
    if not series_list:
        raise ValueError("No return series provided")

    common_dates = set(series_list[0].dates)
    for series in series_list[1:]:
        common_dates &= set(series.dates)

    aligned_dates = sorted(common_dates)
    if not aligned_dates:
        raise ValueError("No overlapping dates among the assets")

    lookup = {series.ticker: series.to_dict() for series in series_list}
    matrix = [
        [lookup[series.ticker][date] for series in series_list]
        for date in aligned_dates
    ]
    return aligned_dates, matrix


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot compute mean of empty sequence")
    return sum(values) / len(values)


def covariance(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Series length mismatch")
    n = len(x)
    if n < 2:
        raise ValueError("At least two observations required for covariance")
    avg_x = mean(x)
    avg_y = mean(y)
    return sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x, y)) / (n - 1)


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vector)) for row in matrix]


def vecdot(a: List[float], b: List[float]) -> float:
    return sum(i * j for i, j in zip(a, b))


def identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def invert(matrix: List[List[float]]) -> List[List[float]]:
    n = len(matrix)
    aug = [row[:] + identity_row[:] for row, identity_row in zip(matrix, identity(n))]

    for col in range(n):
        pivot = None
        pivot_abs = 0.0
        for row in range(col, n):
            val = abs(aug[row][col])
            if val > pivot_abs and val > 1e-12:
                pivot = row
                pivot_abs = val
        if pivot is None:
            raise ValueError("Matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        aug[col] = [value / pivot_val for value in aug[col]]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [val - factor * ref for val, ref in zip(aug[row], aug[col])]

    inverse = [row[n:] for row in aug]
    return inverse


###############################################################################
# Efficient frontier calculations
###############################################################################


def compute_statistics(matrix: List[List[float]]) -> Tuple[List[float], List[List[float]], List[float]]:
    columns = transpose(matrix)
    means = [mean(col) for col in columns]
    cov = [
        [covariance(columns[i], columns[j]) for j in range(len(columns))]
        for i in range(len(columns))
    ]
    vols = [sqrt(cov[i][i]) for i in range(len(columns))]
    return means, cov, vols


@dataclass
class FrontierPoint:
    expected_return: float
    volatility: float
    weights: List[float]


@dataclass
class EfficientFrontierResult:
    tickers: List[str]
    mean_returns: List[float]
    volatilities: List[float]
    covariance: List[List[float]]
    frontier: List[FrontierPoint]
    tangency: FrontierPoint
    risk_free_rate: float
    observations: int
    trading_days: int


class EfficientFrontier:
    def __init__(self, mean_returns: List[float], cov_matrix: List[List[float]]):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.inv_cov = invert(cov_matrix)

    def min_variance_weights(self, target_return: float) -> List[float]:
        ones = [1.0] * len(self.mean_returns)
        mean_vec = self.mean_returns

        inv_cov_mu = matvec(self.inv_cov, mean_vec)
        inv_cov_ones = matvec(self.inv_cov, ones)

        a = vecdot(mean_vec, inv_cov_mu)
        b = vecdot(mean_vec, inv_cov_ones)
        c = vecdot(ones, inv_cov_ones)

        det = a * c - b * b
        if abs(det) <= 1e-16:
            raise ValueError("Ill-conditioned covariance matrix")

        lambda1 = (c * target_return - b) / det
        lambda2 = (a - b * target_return) / det
        return [lambda1 * w_mu + lambda2 * w_ones for w_mu, w_ones in zip(inv_cov_mu, inv_cov_ones)]

    def portfolio_stats(self, weights: List[float]) -> Tuple[float, float]:
        mean_return = vecdot(weights, self.mean_returns)
        variance = vecdot(weights, matvec(self.cov_matrix, weights))
        return mean_return, sqrt(variance)

    def efficient_frontier(self, steps: int) -> List[FrontierPoint]:
        min_return = min(self.mean_returns)
        max_return = max(self.mean_returns)
        target_returns = [
            min_return + (max_return - min_return) * i / (steps - 1)
            for i in range(steps)
        ]
        points = []
        for target in target_returns:
            weights = self.min_variance_weights(target)
            exp_return, volatility = self.portfolio_stats(weights)
            points.append(FrontierPoint(exp_return, volatility, weights))
        return points

    def max_sharpe(self, risk_free_rate: float) -> FrontierPoint:
        excess = [mu - risk_free_rate for mu in self.mean_returns]
        inv_cov_excess = matvec(self.inv_cov, excess)
        denom = vecdot([1.0] * len(self.mean_returns), inv_cov_excess)
        if abs(denom) <= 1e-16:
            raise ValueError("Cannot compute tangency portfolio")
        weights = [value / denom for value in inv_cov_excess]
        exp_return, volatility = self.portfolio_stats(weights)
        return FrontierPoint(exp_return, volatility, weights)


###############################################################################
# SVG rendering
###############################################################################


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_svg(
    tickers: Sequence[str],
    stats: EfficientFrontierResult,
    width: int = 720,
    height: int = 480,
) -> str:
    padding = 60
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding

    max_vol = max(point.volatility for point in stats.frontier + [stats.tangency]) or 1e-12
    max_ret = max(point.expected_return for point in stats.frontier + [stats.tangency]) or 1e-12

    def scale_x(volatility: float) -> float:
        return padding + (volatility / max_vol) * chart_width

    def scale_y(return_: float) -> float:
        return padding + chart_height - (return_ / max_ret) * chart_height

    frontier_path = " ".join(
        f"L {scale_x(pt.volatility):.2f} {scale_y(pt.expected_return):.2f}"
        for pt in stats.frontier[1:]
    )
    start = stats.frontier[0]
    path_d = f"M {scale_x(start.volatility):.2f} {scale_y(start.expected_return):.2f} {frontier_path}"

    asset_points = []
    for ticker, mean_return, vol in zip(
        stats.tickers, stats.mean_returns, stats.volatilities
    ):
        asset_points.append(
            f'<circle cx="{scale_x(vol):.2f}" cy="{scale_y(mean_return):.2f}" r="5" fill="#1f77b4" />'
            f"<text x='{scale_x(vol) + 8:.2f}' y='{scale_y(mean_return) - 8:.2f}' font-size='14'>{ticker}</text>"
        )

    tangency = stats.tangency
    tangency_circle = (
        f'<circle cx="{scale_x(tangency.volatility):.2f}" cy="{scale_y(tangency.expected_return):.2f}" '
        f'r="6" fill="#d62728" />'
        f"<text x='{scale_x(tangency.volatility) + 10:.2f}' y='{scale_y(tangency.expected_return) + 4:.2f}' font-size='14'>Max Sharpe</text>"
    )

    annotations = []
    annotations.append(
        f"<text x='{padding}' y='{padding - 20}' font-size='18' font-weight='bold'>Efficient Frontier</text>"
    )
    annotations.append(
        f"<text x='{padding}' y='{height - padding + 40}' font-size='12'>Risk-free rate: {format_percentage(stats.risk_free_rate)}</text>"
    )
    annotations.append(
        f"<text x='{padding}' y='{height - padding + 60}' font-size='12'>Annualisation: {stats.trading_days} trading days</text>"
    )

    for idx, (ticker, mean_return, vol) in enumerate(
        zip(stats.tickers, stats.mean_returns, stats.volatilities)
    ):
        y = padding + idx * 20
        annotations.append(
            f"<text x='{width - padding - 220}' y='{y + 20}' font-size='12'>{ticker}: {format_percentage(mean_return)} / {format_percentage(vol)}</text>"
        )

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' stroke='#dddddd'/>
  <line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='#333333' stroke-width='2'/>
  <line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='#333333' stroke-width='2'/>
  <path d='{path_d}' fill='none' stroke='#2ca02c' stroke-width='3'/>
  {''.join(asset_points)}
  {tangency_circle}
  {''.join(annotations)}
</svg>
""".strip()
    return svg


###############################################################################
# CLI interface
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efficient frontier analysis for temp.csv")
    parser.add_argument("--csv", type=Path, default=Path("temp.csv"), help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("assets/efficient_frontier.svg"), help="Output SVG path")
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Annualized risk-free rate expressed as a decimal (e.g. 0.03)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of samples along the efficient frontier",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Number of observations per year used for annualisation",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> EfficientFrontierResult:
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 to form a frontier")
    if args.trading_days <= 0:
        raise ValueError("--trading-days must be positive")

    series = load_close_returns(args.csv)
    tickers = [series_item.ticker for series_item in series]
    _, matrix = align_returns(series)
    mean_returns_daily, cov_matrix_daily, _ = compute_statistics(matrix)
    trading_days = args.trading_days
    mean_returns = [mu * trading_days for mu in mean_returns_daily]
    cov_matrix = [
        [cov * trading_days for cov in row]
        for row in cov_matrix_daily
    ]
    volatilities = [sqrt(cov_matrix[i][i]) for i in range(len(cov_matrix))]

    optimizer = EfficientFrontier(mean_returns, cov_matrix)
    frontier_points = optimizer.efficient_frontier(args.steps)
    tangency = optimizer.max_sharpe(args.risk_free)

    return EfficientFrontierResult(
        tickers=tickers,
        mean_returns=mean_returns,
        volatilities=volatilities,
        covariance=cov_matrix,
        frontier=frontier_points,
        tangency=tangency,
        risk_free_rate=args.risk_free,
        observations=len(matrix),
        trading_days=trading_days,
    )


def save_svg(svg: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run(args)

    print(f"Aligned observations: {result.observations}")
    print(f"Frontier resolution: {len(result.frontier)} points")
    print(f"Annualisation factor: {result.trading_days} trading days")
    print("Annualized mean returns (%):")
    for ticker, mean_return in zip(result.tickers, result.mean_returns):
        print(f"  {ticker}: {mean_return * 100:.2f}")
    print("Annualized volatility (%):")
    for ticker, vol in zip(result.tickers, result.volatilities):
        print(f"  {ticker}: {vol * 100:.2f}")

    tangency = result.tangency
    print("\nMax Sharpe (tangency) portfolio weights:")
    for ticker, weight in zip(result.tickers, tangency.weights):
        print(f"  {ticker}: {weight * 100:.2f}%")
    print(
        f"Expected return: {tangency.expected_return * 100:.2f}% | Volatility: {tangency.volatility * 100:.2f}%"
    )

    svg = make_svg(result.tickers, result)
    save_svg(svg, args.output)
    print(f"SVG chart saved to {args.output}")


if __name__ == "__main__":
    main()
