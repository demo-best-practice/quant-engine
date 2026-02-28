import os
import argparse
import yfinance as yf
import finnhub
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 配置说明: MIN_MCAP 单位为"亿"(美元)
CONFIG = {
    "SCAN_SIZE": 100,
    "REPORT_COUNT": 10,
    
    "MIN_MCAP": 20,           # 最小市值(亿)
    "MIN_TURNOVER_30D": 50,  # 30日成交额(百万美元)
    
    "MIN_ROE": 0.15,          # ROE≥15%，强护城河
    "MAX_PE": 25,
    "MAX_DEBT_EQUITY": 150,   # 负债率≤150%，避免黑天鹅
    "FCF_POSITIVE": False,     # 现金流为正
    
    "Z_LIMIT": -0.8,          # Z-Score阈值
    "RSI_OVERSOLD": 30,      # RSI超卖阈值
    "LOOKBACK_DAYS": 250,
    
    "PREDICT_STEPS": 10,
    "ATR_STOP_MULT": 2.2,
    "EARNINGS_BUFFER": 3
}

class QuantDataFetcher:
    """数据获取器"""
    
    def __init__(self):
        self.cache = {}
        self.sector_pe_cache = {}
    
    def get_info(self, symbol):
        """获取股票info数据"""
        if symbol in self.cache:
            return self.cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self.cache[symbol] = info
            return info
        except Exception as e:
            return {}
    
    def get_financials(self, symbol):
        """获取财务报表数据"""
        try:
            ticker = yf.Ticker(symbol)
            return {
                'income': ticker.income_statement,
                'balance': ticker.balance_sheet,
                'cashflow': ticker.cashflow,
                'info': ticker.info
            }
        except Exception as e:
            return None
    
    def get_sector_pe(self, symbol):
        """获取行业PE中值"""
        if symbol in self.sector_pe_cache:
            return self.sector_pe_cache[symbol]
        
        try:
            info = self.get_info(symbol)
            sector = info.get('sector')
            industry = info.get('industry')
            
            if not sector:
                return None
            
            return {
                'sector': sector,
                'industry': industry,
                'pe': info.get('trailingPE') or info.get('forwardPE')
            }
        except:
            return None

class QuantEngine:
    def __init__(self, config, api_key=None):
        self.cfg = config
        if not api_key:
            api_key = os.environ.get('FINNHUB_API_KEY', '')
        if not api_key:
            raise ValueError("Finnhub API key is required. Set via --api-key argument or FINNHUB_API_KEY environment variable")
        self.fc = finnhub.Client(api_key=api_key)
        self.fetcher = QuantDataFetcher()
        self.sector_pe_map = {}
    
    def _get_clean_symbols(self):
        """清洗代码，确保识别美股主板标的"""
        print("📡 正在同步高质量标的名单...")
        try:
            all_s = self.fc.stock_symbols('US')
            clean = [s['symbol'] for s in all_s if s['symbol'].isalpha() and len(s['symbol']) <= 4]
            return clean[:self.cfg["SCAN_SIZE"]]
        except Exception as e:
            print(f"❌ 获取名单失败: {e}")
            return []
    
    def _update_benchmarks(self, symbols):
        """更新行业PE中值"""
        print("📊 正在计算行业PE中值...")
        
        sector_data = {}
        
        for s in symbols[:30]:
            try:
                info = self.fetcher.get_info(s)
                sector = info.get('sector')
                pe = info.get('trailingPE') or info.get('forwardPE')
                
                if sector and pe and 0 < pe < 100:
                    if sector not in sector_data:
                        sector_data[sector] = []
                    sector_data[sector].append(pe)
            except:
                continue
        
        for sector, pes in sector_data.items():
            if pes:
                self.sector_pe_map[sector] = np.median(pes)
        
        print(f"  已获取 {len(self.sector_pe_map)} 个行业的PE中值")
        if self.sector_pe_map:
            print(f"  行业列表: {list(self.sector_pe_map.keys())[:5]}...")
    
    def run(self):
        start_time = datetime.now()
        
        # 获取股票列表
        symbol_list = self._get_clean_symbols()
        if not symbol_list:
            return
        
        # 阶段一：技术面筛选 (yfinance)
        print(f"⚡ 正在分析 {len(symbol_list)} 只股票的技术面形态...")
        
        tech_hits = []
        batch_size = 25
        
        for i in range(0, len(symbol_list), batch_size):
            batch = symbol_list[i:i + batch_size]
            try:
                data = yf.download(batch, period="400d", group_by='ticker', 
                                   threads=True, progress=False, timeout=15)
                
                for sym in batch:
                    if sym not in data or data[sym].empty:
                        continue
                    
                    df = data[sym].dropna()
                    if len(df) < self.cfg["LOOKBACK_DAYS"]:
                        continue
                    
                    close_series = df['Close']
                    last_price = close_series.iloc[-1]
                    
                    subset = close_series.tail(self.cfg["LOOKBACK_DAYS"])
                    std_dev = subset.std()
                    if pd.isna(std_dev) or std_dev == 0:
                        continue
                    
                    z_score = (last_price - subset.mean()) / std_dev
                    
                    # 计算RSI
                    delta = close_series.diff()
                    up = delta.where(delta > 0, 0).tail(14).mean()
                    down = -delta.where(delta < 0, 0).tail(14).mean()
                    rsi = 100 - (100 / (1 + (up / down))) if down != 0 else (100 if up > 0 else 50)
                    
                    # 技术面筛选：Z-Score + RSI + 成交量
                    if z_score < self.cfg["Z_LIMIT"]:
                        if rsi < self.cfg["RSI_OVERSOLD"]:
                            avg_vol_30d = df['Volume'].tail(30).mean()
                            avg_price_30d = close_series.tail(30).mean()
                            turnover_30d = (avg_vol_30d * avg_price_30d) / 1e6
                            
                            if turnover_30d >= self.cfg["MIN_TURNOVER_30D"]:
                                tech_hits.append({
                                    "symbol": sym, 
                                    "z": z_score, 
                                    "rsi": rsi, 
                                    "df": df, 
                                    "price": last_price
                                })
                
                print(f"进度: {min(i+batch_size, len(symbol_list))}/{len(symbol_list)}")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ 批量处理失败: {e}")
                continue
        
        # 阶段二：基本面审计 (yfinance)
        print(f"\n🔍 技术达标 {len(tech_hits)} 只，开始深度审计基本面...")
        
        self._update_benchmarks([h['symbol'] for h in tech_hits])
        
        final_results = []
        watchlist = []
        
        for hit in tqdm(tech_hits, desc="审计进程"):
            s = hit['symbol']
            
            try:
                # 财报日期避险
                e = self.fc.earnings_calendar(
                    _from=datetime.now().strftime('%Y-%m-%d'),
                    to=(datetime.now() + timedelta(days=self.cfg["EARNINGS_BUFFER"])).strftime('%Y-%m-%d'),
                    symbol=s
                )
                if e.get('earningsCalendar'):
                    continue
                
                # 使用yfinance获取基本面数据
                info = self.fetcher.get_info(s)
                
                if not info:
                    watchlist.append({
                        "代码": s,
                        "行业": "-",
                        "当前价": round(hit['price'], 2),
                        "Z-Score": round(hit['z'], 2),
                        "RSI": round(hit['rsi'], 1),
                        "PE": "-",
                        "行业中值": "-",
                        "ROE%": "-",
                        "负债率%": "-",
                        "市值亿": "-",
                        "FCF亿": "-",
                        "不达标原因": "yfinance无数据"
                    })
                    continue
                
                # 市值 - yfinance返回的是美元
                mcap = 0
                
                if info.get('marketCap'):
                    mcap = float(info.get('marketCap'))
                
                if not mcap or mcap <= 0:
                    try:
                        shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                        price = info.get('currentPrice') or info.get('regularMarketPrice')
                        if shares and price:
                            mcap = float(shares) * float(price)
                    except:
                        pass
                
                # 转换为亿（1亿 = 1e8）
                mcap_100m = mcap / 1e8
                
                # 市值过滤（单位：亿）
                if mcap_100m < self.cfg["MIN_MCAP"]:
                    watchlist.append({
                        "代码": s,
                        "行业": f"{info.get('sector', 'Unknown')}/{info.get('industry', 'Unknown')}",
                        "当前价": round(hit['price'], 2),
                        "Z-Score": round(hit['z'], 2),
                        "RSI": round(hit['rsi'], 1),
                        "PE": "-",
                        "行业中值": "-",
                        "ROE%": "-",
                        "负债率%": "-",
                        "市值亿": round(mcap_100m, 0) if mcap_100m > 0 else 0,
                        "FCF亿": "-",
                        "不达标原因": f"市值不足({round(mcap_100m, 0)}亿<{self.cfg['MIN_MCAP']}亿)"
                    })
                    continue
                
                # 行业
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                # PE
                pe = info.get('trailingPE') or info.get('forwardPE') or 999
                
                # ROE (yfinance返回的是小数形式，如0.15表示15%)
                roe = info.get('returnOnEquity') or 0
                
                # 负债率
                de = info.get('debtToEquity') or 0
                
                # FCF
                try:
                    ticker = yf.Ticker(s)
                    cashflow = ticker.cashflow
                    if cashflow is not None and not cashflow.empty:
                        if 'Free Cash Flow' in cashflow.index:
                            fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                        else:
                            fcf = 0
                    else:
                        fcf = 0
                except:
                    fcf = 0
                
                # 行业中值PE
                sector_pe = self.sector_pe_map.get(sector, 20)
                
                # 检查是否达标（mcap单位为美元，需要转换为亿）
                pe_pass = pe <= min(self.cfg["MAX_PE"], sector_pe * 1.1)
                roe_pass = roe >= self.cfg["MIN_ROE"]
                mcap_pass = mcap >= self.cfg["MIN_MCAP"] * 1e8  # 亿→美元
                de_pass = de <= self.cfg["MAX_DEBT_EQUITY"]
                fcf_pass = fcf > 0 if self.cfg["FCF_POSITIVE"] else True
                
                if pe_pass and roe_pass and mcap_pass and de_pass and fcf_pass:
                    # 趋势预测
                    try:
                        model = ExponentialSmoothing(hit['df']['Close'].tail(60), trend='add').fit()
                        pred = model.forecast(self.cfg["PREDICT_STEPS"]).iloc[-1]
                    except:
                        pred = hit['price']
                    
                    if pred > hit['price']:
                        atr = (hit['df']['High'] - hit['df']['Low']).tail(14).mean()
                        
                        final_results.append({
                            "日期": datetime.now().strftime('%Y-%m-%d'),
                            "代码": s,
                            "行业": f"{sector}/{industry}",
                            "当前价": round(hit['price'], 2),
                            "Z-Score": round(hit['z'], 2),
                            "RSI": round(hit['rsi'], 1),
                            "PE": round(pe, 1),
                            "行业中值": round(sector_pe, 1),
                            "ROE%": round(roe * 100, 1) if roe else 0,
                            "负债率%": round(de, 1),
                            "市值亿": round(mcap / 1e8, 0),
                            "10D预测": round(pred, 2),
                            "止损位": round(hit['price'] - (atr * self.cfg["ATR_STOP_MULT"]), 2),
                            "空间%": round(((pred/hit['price'])-1)*100, 2)
                        })
                    else:
                        watchlist.append({
                            "代码": s,
                            "行业": f"{sector}/{industry}",
                            "当前价": round(hit['price'], 2),
                            "Z-Score": round(hit['z'], 2),
                            "RSI": round(hit['rsi'], 1),
                            "PE": round(pe, 1),
                            "行业中值": round(sector_pe, 1),
                            "ROE%": round(roe * 100, 1) if roe else 0,
                            "负债率%": round(de, 1),
                            "市值亿": round(mcap / 1e8, 0),
                            "FCF亿": round(fcf / 1e9, 1) if fcf else 0,
                            "不达标原因": "预测不上涨"
                        })
                else:
                    reasons = []
                    if not pe_pass:
                        reasons.append(f"PE过高({round(pe,1)})")
                    if not roe_pass:
                        reasons.append(f"ROE过低({round(roe*100,1)}%)")
                    if not de_pass:
                        reasons.append(f"负债率过高({round(de,1)}%)")
                    if not fcf_pass:
                        reasons.append("FCF为负")
                    
                    watchlist.append({
                        "代码": s,
                        "行业": f"{sector}/{industry}",
                        "当前价": round(hit['price'], 2),
                        "Z-Score": round(hit['z'], 2),
                        "RSI": round(hit['rsi'], 1),
                        "PE": round(pe, 1),
                        "行业中值": round(sector_pe, 1),
                        "ROE%": round(roe * 100, 1) if roe else 0,
                        "负债率%": round(de, 1),
                        "市值亿": round(mcap / 1e8, 0),
                        "FCF亿": round(fcf / 1e9, 1) if fcf else 0,
                        "不达标原因": "; ".join(reasons) if reasons else "其他"
                    })
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"\n❌ 处理 {s} 出错: {e}")
                continue
        
        # 输出结果
        if final_results:
            df_final = pd.DataFrame(final_results).sort_values("空间%", ascending=False).head(self.cfg["REPORT_COUNT"])
            fname = f"波段套利报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
            df_final.to_excel(fname, index=False)
            print(f"\n✅ 扫描成功！耗时: {datetime.now()-start_time}")
            print(f"📂 结果已保存至: {fname}")
            print(df_final.to_string(index=False))
        else:
            print("\n" + "=" * 60)
            print("📊 扫描结果分析")
            print("=" * 60)
            print(f"  • 技术面初筛达标: {len(tech_hits)} 只")
            print(f"  • 基本面审计通过: {len(final_results)} 只")
            print(f"  • 待观察(有瑕疵): {len(watchlist)} 只")
            print("=" * 60)
            
            if watchlist:
                print(f"\n📋 待观察名单详情 ({len(watchlist)}只):")
                print("-" * 100)
                
                def fmt(val, fmt_str='.1f'):
                    if val == '-' or val is None:
                        return '-'
                    try:
                        return f"{float(val):{fmt_str}}"
                    except:
                        return str(val)
                
                for item in watchlist:
                    pe_val = fmt(item.get('PE'), '6.1f')
                    sector_pe_val = fmt(item.get('行业中值'), '5.1f')
                    roe_val = fmt(item.get('ROE%'), '5.1f')
                    de_val = fmt(item.get('负债率%'), '5.1f')
                    mcap_val = fmt(item.get('市值亿'), '6.0f')
                    
                    print(f"  {item['代码']:6} | 价:{item['当前价']:8.2f} | Z:{item['Z-Score']:6.2f} | RSI:{item['RSI']:5.1f} | PE:{pe_val} | 行业中值:{sector_pe_val} | ROE:{roe_val}% | 负债:{de_val}% | 市值:{mcap_val}亿")
                    print(f"         └─ 不达标原因: {item['不达标原因']}")
                
                print("-" * 100)
                
                watch_fname = f"待观察名单_{datetime.now().strftime('%Y%m%d')}.xlsx"
                df_watch = pd.DataFrame(watchlist)
                df_watch.to_excel(watch_fname, index=False)
                print(f"\n📂 已保存至: {watch_fname}")
                
                # 统计
                print("\n📊 不达标原因统计:")
                reason_counts = {}
                for item in watchlist:
                    reason = item.get('不达标原因', '未知')
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    print(f"   {reason}: {count}只")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quant Engine - Stock Screening')
    
    # API
    parser.add_argument('--api-key', type=str, required=True, help='Finnhub API key')
    
    # 基本面参数
    parser.add_argument('--scan-size', type=int, default=CONFIG['SCAN_SIZE'], help='扫描股票数量')
    parser.add_argument('--report-count', type=int, default=CONFIG['REPORT_COUNT'], help='输出标的数量')
    parser.add_argument('--min-mcap', type=float, default=CONFIG['MIN_MCAP'], help='最小市值(亿)')
    parser.add_argument('--max-pe', type=float, default=CONFIG['MAX_PE'], help='最大PE值')
    parser.add_argument('--min-roe', type=float, default=CONFIG['MIN_ROE'], help='最小ROE')
    parser.add_argument('--max-debt-equity', type=float, default=CONFIG['MAX_DEBT_EQUITY'], help='最大负债率')
    parser.add_argument('--fcf-positive', type=str, default='false', help='要求FCF为正 (true/false)')
    
    # 技术面参数
    parser.add_argument('--z-limit', type=float, default=CONFIG['Z_LIMIT'], help='Z-Score阈值')
    parser.add_argument('--rsi-oversold', type=float, default=CONFIG['RSI_OVERSOLD'], help='RSI超卖阈值')
    parser.add_argument('--min-turnover', type=float, default=CONFIG['MIN_TURNOVER_30D'], help='30日最小成交额(百万美元)')
    parser.add_argument('--lookback-days', type=int, default=CONFIG['LOOKBACK_DAYS'], help='回看天数')
    
    # 预测参数
    parser.add_argument('--predict-steps', type=int, default=CONFIG['PREDICT_STEPS'], help='预测天数')
    parser.add_argument('--atr-stop-mult', type=float, default=CONFIG['ATR_STOP_MULT'], help='ATR止损倍数')
    parser.add_argument('--earnings-buffer', type=int, default=CONFIG['EARNINGS_BUFFER'], help='财报发布日期缓冲天数')
    
    args = parser.parse_args()
    
    # Override config with CLI args
    CONFIG['SCAN_SIZE'] = args.scan_size
    CONFIG['REPORT_COUNT'] = args.report_count
    CONFIG['MIN_MCAP'] = args.min_mcap
    CONFIG['MAX_PE'] = args.max_pe
    CONFIG['MIN_ROE'] = args.min_roe
    CONFIG['MAX_DEBT_EQUITY'] = args.max_debt_equity
    CONFIG['FCF_POSITIVE'] = args.fcf_positive.lower() == 'true'
    CONFIG['Z_LIMIT'] = args.z_limit
    CONFIG['RSI_OVERSOLD'] = args.rsi_oversold
    CONFIG['MIN_TURNOVER_30D'] = args.min_turnover
    CONFIG['LOOKBACK_DAYS'] = args.lookback_days
    CONFIG['PREDICT_STEPS'] = args.predict_steps
    CONFIG['ATR_STOP_MULT'] = args.atr_stop_mult
    CONFIG['EARNINGS_BUFFER'] = args.earnings_buffer
    
    print("🚀 启动量化引擎...")
    print("=" * 50)
    print("📌 当前配置:")
    print(f"   市值≥{CONFIG['MIN_MCAP']}亿 | ROE≥{CONFIG['MIN_ROE']*100}% | PE≤{CONFIG['MAX_PE']} | 负债≤{CONFIG['MAX_DEBT_EQUITY']}%")
    print(f"   Z-Score<{CONFIG['Z_LIMIT']} | RSI<{CONFIG['RSI_OVERSOLD']} | 成交额≥{CONFIG['MIN_TURNOVER_30D']}M")
    print("=" * 50)
    
    engine = QuantEngine(CONFIG, api_key=args.api_key)
    engine.run()
