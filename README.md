# 创业板多因子选股策略

一个用于学习和求职展示的横截面多因子量化研究项目。

## 研究思路

研究对象为创业板股票池，使用日线数据构造：

- Momentum 20：20日价格动量
- Volatility 20：20日收益率波动
- Liquidity 20：20日平均成交额

因子先进行横截面排名，再构造综合评分。

当前核心组合：

- Top 5%
- 等权持仓
- 20个交易日调仓
- 纳入交易成本

## 回测时点

策略采用：

> T日收盘产生信号，T+1开始持仓。

回测中使用T日收盘价作为T+1收益计算的基准，避免因为截取持仓区间而遗漏第一天收益。

## 项目结构

```text
growth-factor-strategy/
├── growth_factor_strategy.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── growth_daily/
```

## 数据

将股票日线 parquet 文件放入：

```text
data/growth_daily/
```

程序默认读取其中全部 `.parquet` 文件。

建议 GitHub 不直接提交完整历史行情数据，可使用 `.gitignore` 忽略数据文件。

## 运行

```bash
pip install -r requirements.txt
python growth_factor_strategy.py
```

运行后结果会保存到：

```text
output/
```

包括策略净值 CSV 和净值图。

## 注意

这是研究和学习项目，不构成投资建议。
