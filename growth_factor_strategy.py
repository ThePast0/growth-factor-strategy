"""
创业板多因子选股策略
====================

研究流程：
1. 读取日线数据
2. 计算 Momentum / Volatility / Liquidity
3. 横截面标准化
4. 多因子选股
5. 组合回测
6. 稳健性与样本外检验
7. 输出结果和图表

说明：
本项目用于量化研究与学习，不构成投资建议。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置 Matplotlib 中文字体
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "growth_daily"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_PCT = 0.05
REBALANCE_PERIOD = 20
TRANSACTION_COST = 0.001

DATA_DIR = "/Users/lucas/量化/data/growth_daily"
def load_data(data_dir=DATA_DIR):
    """读取股票日线 parquet 数据。"""
    files = sorted(Path(data_dir).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"没有找到 parquet 文件，请把股票数据放到：{data_dir}"
        )

    frames = []
    for file in files:
        temp = pd.read_parquet(file)
        frames.append(temp)

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    return df


def calculate_factors(df):
    """计算三个基础因子。"""

    data = df.copy()

    # Momentum：过去20个交易日价格变化
    data["momentum_20"] = (
        data.groupby("code")["close"].pct_change(20)
    )

    # Volatility：过去20日收益率波动
    daily_return = data.groupby("code")["close"].pct_change()

    data["volatility_20"] = (
        daily_return
        .groupby(data["code"])
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Liquidity：过去20日成交额平均值
    data["liquidity_20"] = (
        data.groupby("code")["amount"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return data


def cross_section_rank(df, factor):
    """按照交易日进行横截面排名，结果范围为0~1。"""
    return df.groupby("date")[factor].rank(pct=True)


def prepare_factors(df):
    """计算因子排名并构造最终评分。"""

    data = calculate_factors(df)

    factors = ["momentum_20", "volatility_20", "liquidity_20"]

    for factor in factors:
        data[f"{factor}_rank"] = cross_section_rank(data, factor)

    # 当前研究阶段使用等权综合评分。
    # 对负向因子先反向处理，使得分越高代表预期越优。
    data["factor_score"] = (
        data["momentum_20_rank"]
        + (1 - data["volatility_20_rank"])
        + (1 - data["liquidity_20_rank"])
    ) / 3

    return data


def select_portfolio(signal_data, top_pct=TOP_PCT):
    """根据因子评分选择每个调仓日的 Top 股票。"""

    data = signal_data.dropna(
        subset=["factor_score", "close"]
    ).copy()

    selected = {}

    for date, group in data.groupby("date", sort=True):
        n = max(1, int(len(group) * top_pct))

        stocks = (
            group.sort_values("factor_score", ascending=False)
            .head(n)["code"]
            .tolist()
        )

        selected[date] = stocks

    return selected


def backtest_factor_strategy(
    df,
    top_pct=TOP_PCT,
    rebalance_period=REBALANCE_PERIOD,
    transaction_cost=TRANSACTION_COST,
):
    """
    严格按照 T 日信号、T+1 开始持仓进行回测。

    T 日：
        使用当日收盘数据生成组合。

    T+1：
        开始承担组合收益。

    每隔 rebalance_period 个交易日重新选股。
    """

    data = df.copy()
    data = data.sort_values(["date", "code"]).reset_index(drop=True)

    dates = sorted(data["date"].unique())

    # 至少需要下一交易日才能开始回测
    rebalance_dates = dates[:-1:rebalance_period]

    portfolio_returns = []
    previous_weights = {}

    for i, signal_date in enumerate(rebalance_dates):

        next_index = dates.index(signal_date) + 1

        if next_index >= len(dates):
            break

        start_date = dates[next_index]

        if i + 1 < len(rebalance_dates):
            next_signal = rebalance_dates[i + 1]
            holding_dates = [
                d for d in dates
                if start_date <= d < next_signal
            ]
        else:
            holding_dates = [
                d for d in dates
                if d >= start_date
            ]

        signal_data = data[data["date"] == signal_date]

        signal_data = signal_data.dropna(
            subset=["factor_score"]
        )

        if signal_data.empty:
            continue

        n = max(1, int(len(signal_data) * top_pct))

        selected = (
            signal_data
            .sort_values("factor_score", ascending=False)
            .head(n)["code"]
            .tolist()
        )

        if not selected:
            continue

        # 每只股票等权
        weight = 1 / len(selected)
        current_weights = {code: weight for code in selected}

        # 换手率 = 新旧组合权重变化绝对值之和
        all_codes = set(previous_weights) | set(current_weights)

        turnover = sum(
            abs(
                current_weights.get(code, 0)
                - previous_weights.get(code, 0)
            )
            for code in all_codes
        )

        # 取信号日前一个交易日的收盘价作为收益计算基准。
        # 这样第一天收益能够正确计算 T -> T+1。
        previous_date = signal_date

        price_data = data[
            data["code"].isin(selected)
            & data["date"].isin(
                [previous_date] + holding_dates
            )
        ][["date", "code", "close"]].copy()

        price_data = price_data.sort_values(["code", "date"])

        price_data["stock_return"] = (
            price_data
            .groupby("code")["close"]
            .pct_change()
        )

        holding = price_data[
            price_data["date"].isin(holding_dates)
        ].copy()

        # 每只股票等权，因此直接计算横截面平均收益。
        daily_returns = (
            holding
            .groupby("date")["stock_return"]
            .mean()
            .dropna()
        )

        if daily_returns.empty:
            continue

        # 交易成本在调仓后第一天统一扣除。
        if turnover > 0:
            first_day = daily_returns.index[0]
            daily_returns.loc[first_day] -= (
                turnover * transaction_cost
            )

        for date, ret in daily_returns.items():
            portfolio_returns.append(
                {
                    "date": date,
                    "return": ret,
                }
            )

        previous_weights = current_weights

    result = pd.DataFrame(portfolio_returns)

    if result.empty:
        raise ValueError("回测没有产生有效收益数据。")

    result = (
        result
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )

    result["nav"] = (1 + result["return"]).cumprod()

    daily_std = result["return"].std()

    if daily_std == 0 or pd.isna(daily_std):
        sharpe = np.nan
    else:
        sharpe = (
            result["return"].mean()
            / daily_std
            * np.sqrt(252)
        )

    years = len(result) / 252

    total_return = result["nav"].iloc[-1] - 1

    annual_return = (
        result["nav"].iloc[-1] ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    drawdown = (
        result["nav"]
        / result["nav"].cummax()
        - 1
    )

    metrics = {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "最大回撤": drawdown.min(),
        "Sharpe": sharpe,
        "胜率": (result["return"] > 0).mean(),
        "交易日": len(result),
    }

    return result, metrics


def plot_nav(result, title="策略净值"):
    """绘制策略净值曲线。"""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(result.index, result["nav"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "strategy_nav.png",
        dpi=150,
    )
    plt.show()


def main():
    print("=" * 70)
    print("创业板多因子策略")
    print("=" * 70)

    print("\n正在读取数据...")
    df = load_data()

    print("数据行数：", len(df))
    print("股票数量：", df["code"].nunique())
    print(
        "日期范围：",
        df["date"].min(),
        "→",
        df["date"].max(),
    )

    print("\n正在计算因子...")
    df = prepare_factors(df)

    print("\n正在进行回测...")
    result, metrics = backtest_factor_strategy(
        df,
        top_pct=TOP_PCT,
        rebalance_period=REBALANCE_PERIOD,
        transaction_cost=TRANSACTION_COST,
    )

    print("\n" + "=" * 70)
    print("回测结果")
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    result.to_csv(
        OUTPUT_DIR / "strategy_nav.csv",
        encoding="utf-8-sig",
    )

    plot_nav(result)

    print("\n结果已保存到 output/ 目录。")




# ============================================================
# 基准比较
# ============================================================

def calculate_equal_weight_benchmark(df, start_date, end_date):
    """计算创业板股票池等权基准。"""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[
        (data["date"] >= pd.to_datetime(start_date))
        & (data["date"] <= pd.to_datetime(end_date))
    ].copy()

    data["daily_return"] = data.groupby("code")["close"].pct_change()

    benchmark_return = (
        data.groupby("date")["daily_return"]
        .mean()
        .dropna()
    )

    benchmark = pd.DataFrame({"return": benchmark_return})
    benchmark["nav"] = (1 + benchmark["return"]).cumprod()
    return benchmark


def calculate_performance(nav):
    """根据净值序列计算总收益、年化收益、回撤和 Sharpe。"""
    nav = nav.dropna()
    daily_return = nav.pct_change().dropna()

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1

    if daily_return.std() == 0 or pd.isna(daily_return.std()):
        sharpe = np.nan
    else:
        sharpe = daily_return.mean() / daily_return.std() * np.sqrt(252)

    drawdown = nav / nav.cummax() - 1

    return {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "最大回撤": drawdown.min(),
        "Sharpe": sharpe,
    }


def plot_strategy_vs_benchmark(comparison, output_file):
    """绘制策略与创业板等权基准净值曲线。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(comparison.index, comparison["strategy"], label="Strategy")
    ax.plot(comparison.index, comparison["benchmark"], label="Equal-weight Benchmark")
    ax.set_title("Strategy vs ChiNext Equal-weight Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


# ============================================================
# 完整运行入口
# ============================================================

def run_strategy():
    """运行策略、基准比较并保存结果。"""

    print("=" * 70)
    print("创业板多因子策略")
    print("=" * 70)

    print("\n正在读取数据...")
    df = load_data()

    print("数据行数：", len(df))
    print("股票数量：", df["code"].nunique())
    print("日期范围：", df["date"].min(), "→", df["date"].max())

    print("\n正在计算因子...")
    df = prepare_factors(df)

    print("\n正在进行回测...")
    result, metrics = backtest_factor_strategy(
        df,
        top_pct=TOP_PCT,
        rebalance_period=REBALANCE_PERIOD,
        transaction_cost=TRANSACTION_COST,
    )

    print("\n" + "=" * 70)
    print("回测结果")
    print("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    result.to_csv(OUTPUT_DIR / "strategy_nav.csv", encoding="utf-8-sig")
    plot_nav(result)

    start_date = result.index.min()
    end_date = result.index.max()
    benchmark = calculate_equal_weight_benchmark(df, start_date, end_date)

    comparison = pd.DataFrame({
        "strategy": result["nav"],
        "benchmark": benchmark["nav"],
    }).dropna()

    comparison["strategy"] /= comparison["strategy"].iloc[0]
    comparison["benchmark"] /= comparison["benchmark"].iloc[0]

    strategy_metrics = calculate_performance(comparison["strategy"])
    benchmark_metrics = calculate_performance(comparison["benchmark"])

    print("\n" + "=" * 70)
    print("策略 vs 创业板股票池等权基准")
    print("=" * 70)
    print(f"策略总收益率：{strategy_metrics['总收益率']:.2%}")
    print(f"基准总收益率：{benchmark_metrics['总收益率']:.2%}")
    print(f"策略年化收益率：{strategy_metrics['年化收益率']:.2%}")
    print(f"基准年化收益率：{benchmark_metrics['年化收益率']:.2%}")
    print(f"策略最大回撤：{strategy_metrics['最大回撤']:.2%}")
    print(f"基准最大回撤：{benchmark_metrics['最大回撤']:.2%}")
    print(f"策略 Sharpe：{strategy_metrics['Sharpe']:.3f}")
    print(f"基准 Sharpe：{benchmark_metrics['Sharpe']:.3f}")
    print(f"策略相对基准的总收益差：{strategy_metrics['总收益率'] - benchmark_metrics['总收益率']:.2%}")
    print(f"最终策略净值：{comparison['strategy'].iloc[-1]:.3f}")
    print(f"最终基准净值：{comparison['benchmark'].iloc[-1]:.3f}")

    comparison.to_csv(OUTPUT_DIR / "strategy_vs_benchmark.csv", encoding="utf-8-sig")
    plot_strategy_vs_benchmark(comparison, OUTPUT_DIR / "strategy_vs_benchmark.png")

    print("\n结果已保存到：")
    print(OUTPUT_DIR)

    return result, metrics, comparison


if __name__ == "__main__":
    run_strategy()
