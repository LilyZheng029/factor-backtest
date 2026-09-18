"""
A股反转因子回测脚本
作者：郑景元
日期：2026-09
说明：验证20日反转因子在沪深300成分股中的选股能力
"""

# ========== 1. 导入库 ==========
import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 2. 配置参数 ==========
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'
STOCK_NUM = 50
FACTOR_WINDOW = 20
FORWARD_WINDOW = 5
GROUP_NUM = 5


# ========== 3. 函数定义 ==========
def login_baostock():
    lg = bs.login()
    print(f"登录状态: {lg.error_code} - {lg.error_msg}")


def get_stock_list(num):
    rs = bs.query_hs300_stocks()
    hs300 = rs.get_data()
    stock_list = hs300['code'].tolist()[:num]
    print(f"共获取 {len(stock_list)} 只成分股")
    return stock_list


def get_price_data(stock_list, start_date, end_date):
    all_data = []
    for code in stock_list:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        df = rs.get_data()
        if len(df) > 0:
            all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data['close'] = data['close'].astype(float)
    data = data.sort_values(['code', 'date'])
    print(f"总数据量: {len(data)} 行")
    return data


def calculate_factor(data, factor_window, forward_window):
    data['ret_20d'] = data.groupby('code')['close'].pct_change(factor_window)
    data['reversal_factor'] = -data['ret_20d']
    data['forward_ret_5d'] = data.groupby('code')['close'].pct_change(forward_window).shift(-forward_window)
    factor_data = data.dropna(subset=['reversal_factor', 'forward_ret_5d']).copy()
    print(f"有效数据量: {len(factor_data)} 行")
    return factor_data


def group_backtest(factor_data, group_num):
    factor_data['group'] = factor_data.groupby('date')['reversal_factor'].transform(
        lambda x: pd.qcut(x, group_num, labels=False, duplicates='drop')
    )
    group_ret = factor_data.groupby(['date', 'group'])['forward_ret_5d'].mean().unstack()
    group_nav = (1 + group_ret).cumprod()
    group_ret['long_short'] = group_ret[group_num - 1] - group_ret[0]
    ls_nav = (1 + group_ret['long_short']).cumprod()
    print("分组回测完成")
    return group_nav, ls_nav


def calculate_ic(factor_data):
    ic_series = factor_data.groupby('date').apply(
        lambda x: x['reversal_factor'].corr(x['forward_ret_5d'])
    )
    print(f"IC均值: {ic_series.mean():.4f}")
    print(f"IC标准差: {ic_series.std():.4f}")
    print(f"ICIR: {ic_series.mean()/ic_series.std():.4f}")
    return ic_series


def plot_results(group_nav, ls_nav, ic_series, group_num):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for g in range(group_num):
        axes[0].plot(group_nav.index, group_nav[g], label=f'Group {g+1}')
    axes[0].plot(ls_nav.index, ls_nav, label='Long-Short', linewidth=2, linestyle='--', color='black')
    axes[0].legend()
    axes[0].set_title('分组累计净值 & 多空组合')
    axes[0].set_ylabel('累计净值')

    axes[1].bar(ic_series.index, ic_series.values, width=1, color='steelblue')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title(f'IC时序 (均值={ic_series.mean():.4f})')
    axes[1].set_ylabel('IC')

    plt.tight_layout()
    plt.savefig('factor_report.png', dpi=150)
    plt.show()
    print("图表已保存为 factor_report.png")


# ========== 4. 主程序入口 ==========
if __name__ == '__main__':
    login_baostock()
    stock_list = get_stock_list(STOCK_NUM)
    data = get_price_data(stock_list, START_DATE, END_DATE)
    factor_data = calculate_factor(data, FACTOR_WINDOW, FORWARD_WINDOW)
    group_nav, ls_nav = group_backtest(factor_data, GROUP_NUM)
    ic_series = calculate_ic(factor_data)
    plot_results(group_nav, ls_nav, ic_series, GROUP_NUM)

    print("=" * 50)
    print(f"IC均值: {ic_series.mean():.4f}")
    print(f"ICIR: {ic_series.mean()/ic_series.std():.4f}")
    print(f"多空组合年化收益: {(ls_nav.iloc[-1]**(252/len(ls_nav))-1)*100:.2f}%")
    print(f"多空组合最大回撤: {(ls_nav/ls_nav.cummax()-1).min()*100:.2f}%")
    print("=" * 50)

    bs.logout()
    print("BaoStock已登出")