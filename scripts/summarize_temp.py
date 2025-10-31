import argparse
from pathlib import Path
from csv import reader
from datetime import datetime
from math import sqrt


def stats(values):
    n = len(values)
    avg = sum(values) / n
    var = sum((v - avg) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return avg, sqrt(var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="temp.csv 시계열 지표를 요약합니다")
    parser.add_argument("--csv", type=Path, default=Path("temp.csv"), help="입력 CSV 경로")
    return parser.parse_args()


def main():
    args = parse_args()

    with args.csv.open() as f:
        csv_reader = reader(f)
        header1, header2, header3 = [next(csv_reader) for _ in range(3)]
        rows = list(csv_reader)

    metrics = ["Close", "High", "Low", "Open", "Volume"]
    indices = {metric: {} for metric in metrics}
    for idx, (metric_name, ticker) in enumerate(zip(header1[1:], header2[1:]), start=1):
        if metric_name in metrics:
            indices[metric_name][ticker] = idx

    series = {}
    for row in rows:
        date = datetime.strptime(row[0], "%Y-%m-%d").date()
        for metric in metrics:
            for ticker, col in indices[metric].items():
                value = row[col]
                if value:
                    series.setdefault(ticker, {}).setdefault(metric, []).append((date, float(value)))

    for ticker, metric_data in series.items():
        closes = sorted(metric_data["Close"])
        volumes = sorted(metric_data["Volume"])
        returns = [
            (curr - prev) / prev * 100
            for (_, prev), (_, curr) in zip(closes, closes[1:])
            if prev
        ]

        close_vals = [v for _, v in closes]
        volume_vals = [v for _, v in volumes]

        close_avg, close_std = stats(close_vals)
        volume_avg, volume_std = stats(volume_vals)
        return_avg, return_std = stats(returns)

        print(f"{ticker} | 관측일수 {len(close_vals)}일 | {closes[0][0]} ~ {closes[-1][0]}")
        print(
            "  종가: "
            f"최신={close_vals[-1]:.2f}, "
            f"평균={close_avg:.2f}±{close_std:.2f}, "
            f"최저={min(close_vals):.2f}, 최고={max(close_vals):.2f}"
        )
        print(
            "  거래량(백만 주): "
            f"평균={volume_avg/1e6:.2f}M±{volume_std/1e6:.2f}M, "
            f"최저={min(volume_vals)/1e6:.2f}M, 최고={max(volume_vals)/1e6:.2f}M"
        )
        print(
            "  일간 수익률(%): "
            f"평균={return_avg:.2f}%, 표준편차={return_std:.2f}%, "
            f"최저={min(returns):.2f}%, 최고={max(returns):.2f}%\n"
        )


if __name__ == "__main__":
    main()
