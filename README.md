# Quant Engine

美股量化选股引擎，基于技术面和基本面筛选优质标的。

## 功能特点

- **数据来源**: Finnhub（股票列表）+ yfinance（行情和基本面）
- **筛选流程**: 技术面初筛 → 基本面审计 → 趋势预测

## 安装依赖

```bash
pip install yfinance finnhub pandas numpy tqdm statsmodels
```


## 使用方法

### 基本用法

```bash
python quant_engine.py --api-key YOUR_FINNHUB_API_KEY
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--api-key` | Finnhub API Key | **必填** |
| `--scan-size` | 扫描股票数量 | 100 |
| `--report-count` | 输出标的数量 | 10 |
| `--min-mcap` | 最小市值(亿) | 20 |
| `--max-pe` | 最大PE值 | 25 |
| `--min-roe` | 最小ROE | 0.15 |
| `--max-debt-equity` | 最大负债率(%) | 150 |
| `--fcf-positive` | 要求FCF为正(true/false) | false |
| `--z-limit` | Z-Score阈值(负值) | -0.8 |
| `--rsi-oversold` | RSI超卖阈值 | 30 |
| `--min-turnover` | 30日最小成交额(百万美元) | 50 |
| `--lookback-days` | 回看天数 | 250 |
| `--predict-steps` | 预测天数 | 10 |
| `--atr-stop-mult` | ATR止损倍数 | 2.2 |
| `--earnings-buffer` | 财报发布日期缓冲天数 | 3 |

### 完整示例

```bash
python quant_engine.py \
  --api-key YOUR_FINNHUB_API_KEY \
  --scan-size 300 \
  --report-count 10 \
  --min-mcap 20 \
  --max-pe 20 \
  --min-roe 0.15 \
  --max-debt-equity 150 \
  --z-limit -0.8 \
  --rsi-oversold 30

### 环境变量

也可以通过环境变量设置 API Key：

```bash
export FINNHUB_API_KEY=your_api_key
python quant_engine.py
```

## 筛选逻辑

### 阶段一：技术面筛选

| 指标 | 说明 | 默认阈值 |
|------|------|----------|
| Z-Score | 价格偏离度，负值表示低于均值 | < -0.8 |
| RSI | 相对强弱指数，低于30表示超卖 | < 30 |
| RSI | 相对强弱指数，低于50表示超卖 | < 50 |
| 成交量 | 30日成交额 | ≥ 5000万 |

### 阶段二：基本面审计

| 指标 | 说明 | 默认阈值 |
|------|------|----------|
| 市值 | 股票总市值 | ≥ 20亿 |
| ROE | 净资产收益率，越高代表盈利能力越强 | ≥ 15% |
| PE | 市盈率，越低表示估值越便宜 | ≤ 25 |
| 负债率 | 债务/权益比，越低风险越小 | ≤ 150% |
| FCF | 自由现金流 | 可选（默认关闭）|

### 阶段三：趋势预测

使用 Halt-Winters 指数平滑预测未来10日走势，只有预测上涨的标的才会进入最终推荐。

## 推荐参数

### 保守型（推荐）

适合追求稳健的投资者，宁缺毋滥：

```bash
python quant_engine.py --api-key YOUR_KEY \
  --min-mcap 20 \
  --max-pe 20 \
  --min-roe 0.15 \
  --max-debt-equity 150 \
  --fcf-positive false \
  --z-limit -0.8 \
  --rsi-oversold 30
```

参数：
- 市值 ≥ 20亿
- ROE ≥ 15%
- PE ≤ 20
- 负债率 ≤ 150%
- FCF 不强制要求
- Z-Score < -0.8
- RSI < 30

### 平衡型

兼顾收益和风险：

```bash
python quant_engine.py --api-key YOUR_KEY \
  --min-mcap 10 \
  --max-pe 25 \
  --min-roe 0.12 \
  --max-debt-equity 200 \
  --fcf-positive false \
  --z-limit -0.5 \
  --rsi-oversold 40
```

参数：
- 市值 ≥ 10亿
- ROE ≥ 12%
- PE ≤ 25
- 负债率 ≤ 200%
- FCF 不强制要求
- Z-Score < -0.5
- RSI < 40

### 激进型

追求高收益，愿意承担更多风险：

```bash
python quant_engine.py --api-key YOUR_KEY \
  --min-mcap 5 \
  --max-pe 30 \
  --min-roe 0.10 \
  --max-debt-equity 300 \
  --fcf-positive false \
  --z-limit -0.3 \
  --rsi-oversold 50
```

参数：
- 市值 ≥ 20亿
- ROE ≥ 15%
- PE ≤ 20
- 负债率 ≤ 150%
- FCF 不强制要求
- Z-Score < -0.8
- RSI < 50

## 输出结果

程序会生成两个 Excel 文件：

1. `波段套利报告_YYYYMMDD.xlsx` - 符合所有条件的推荐标的
2. `待观察名单_YYYYMMDD.xlsx` - 技术面达标但基本面有瑕疵的股票

## 获取 Finnhub API Key

1. 访问 https://finnhub.io/
2. 注册免费账号
3. 在 Dashboard 获取 API Key
4. 免费版每天支持 60 次 API 调用

## 注意事项

- 首次运行需要下载较多数据，建议设置代理
- yfinance 有频率限制，大量扫描时建议增加间隔
- 筛选结果仅供参考，投资需谨慎
- 建议使用保守型参数进行实盘筛选

## 许可证

MIT License
