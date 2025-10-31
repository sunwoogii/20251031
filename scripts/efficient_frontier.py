from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


###############################################################################
# 데이터 적재 유틸리티
###############################################################################


@dataclass
class ReturnSeries:
    ticker: str
    dates: List[datetime]
    returns: List[float]

    def to_dict(self) -> Dict[datetime, float]:
        return {date: ret for date, ret in zip(self.dates, self.returns)}


def load_close_returns(csv_path: Path) -> List[ReturnSeries]:
    """CSV에 포함된 모든 티커의 일간 종가 기준 수익률을 불러옵니다."""

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
# 선형대수 보조 함수
###############################################################################


def align_returns(series_list: Sequence[ReturnSeries]) -> Tuple[List[datetime], List[List[float]]]:
    if not series_list:
        raise ValueError("수익률 시퀀스가 제공되지 않았습니다")

    common_dates = set(series_list[0].dates)
    for series in series_list[1:]:
        common_dates &= set(series.dates)

    aligned_dates = sorted(common_dates)
    if not aligned_dates:
        raise ValueError("자산 간에 공통 거래일이 존재하지 않습니다")

    lookup = {series.ticker: series.to_dict() for series in series_list}
    matrix = [
        [lookup[series.ticker][date] for series in series_list]
        for date in aligned_dates
    ]
    return aligned_dates, matrix


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("빈 시퀀스의 평균을 계산할 수 없습니다")
    return sum(values) / len(values)


def covariance(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("시퀀스 길이가 일치하지 않습니다")
    n = len(x)
    if n < 2:
        raise ValueError("공분산 계산에는 최소 두 개 이상의 관측치가 필요합니다")
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
            raise ValueError("행렬이 특이행렬입니다")
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
# 효율적 투자선 계산
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
            raise ValueError("공분산 행렬의 조건수가 매우 나빠 최적화를 진행할 수 없습니다")

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
            raise ValueError("접선 포트폴리오를 계산할 수 없습니다")
        weights = [value / denom for value in inv_cov_excess]
        exp_return, volatility = self.portfolio_stats(weights)
        return FrontierPoint(exp_return, volatility, weights)


###############################################################################
# SVG 렌더링
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
        f"<text x='{scale_x(tangency.volatility) + 10:.2f}' y='{scale_y(tangency.expected_return) + 4:.2f}' font-size='14'>최대 샤프</text>"
    )

    annotations = []
    annotations.append(
        f"<text x='{padding}' y='{padding - 20}' font-size='18' font-weight='bold'>효율적 투자선</text>"
    )
    annotations.append(
        f"<text x='{padding}' y='{height - padding + 40}' font-size='12'>무위험 수익률: {format_percentage(stats.risk_free_rate)}</text>"
    )
    annotations.append(
        f"<text x='{padding}' y='{height - padding + 60}' font-size='12'>연환산 기준: 거래일 {stats.trading_days}일</text>"
    )

    for idx, (ticker, mean_return, vol) in enumerate(
        zip(stats.tickers, stats.mean_returns, stats.volatilities)
    ):
        y = padding + idx * 20
        annotations.append(
            f"<text x='{width - padding - 220}' y='{y + 20}' font-size='12'>{ticker}: 기대수익 {format_percentage(mean_return)} / 변동성 {format_percentage(vol)}</text>"
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
    return svg + "\n"


###############################################################################
# CLI 인터페이스
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="temp.csv 데이터를 이용해 효율적 투자선을 계산합니다")
    parser.add_argument("--csv", type=Path, default=Path("temp.csv"), help="입력 CSV 파일 경로")
    parser.add_argument("--output", type=Path, default=Path("assets/efficient_frontier.svg"), help="SVG 결과 저장 경로")
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="연환산 무위험 수익률(예: 0.03)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="효율적 투자선에서 계산할 포인트 개수",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="연환산에 사용할 연간 거래일 수",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> EfficientFrontierResult:
    if args.steps < 2:
        raise ValueError("효율적 투자선을 구성하려면 --steps 값이 2 이상이어야 합니다")
    if args.trading_days <= 0:
        raise ValueError("--trading-days 값은 양수여야 합니다")

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

    print(f"정렬된 관측치 수: {result.observations}")
    print(f"효율적 투자선 분해능: {len(result.frontier)} 포인트")
    print(f"연환산 기준 거래일: {result.trading_days}일")
    print("연환산 기대수익률(%):")
    for ticker, mean_return in zip(result.tickers, result.mean_returns):
        print(f"  {ticker}: {mean_return * 100:.2f}")
    print("연환산 변동성(%):")
    for ticker, vol in zip(result.tickers, result.volatilities):
        print(f"  {ticker}: {vol * 100:.2f}")

    tangency = result.tangency
    print("\n최대 샤프(접선) 포트폴리오 비중:")
    for ticker, weight in zip(result.tickers, tangency.weights):
        print(f"  {ticker}: {weight * 100:.2f}%")
    print(
        f"기대수익률: {tangency.expected_return * 100:.2f}% | 변동성: {tangency.volatility * 100:.2f}%"
    )

    svg = make_svg(result.tickers, result)
    save_svg(svg, args.output)
    print(f"SVG 차트를 {args.output}에 저장했습니다")


if __name__ == "__main__":
    main()
