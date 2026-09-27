# Logic of the strategy
## 1. data
1.1 what is the data we downloaded from the the website:
    'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?field_tdr_date_value=2025&type=daily_treasury_yield_curve'
    we download the yield rate and the yield curve data from day to day.
1.2 what is the yield rate?
    基础产品是美国财政部已经发行、正在二级市场交易的国债：
    - Treasury Bills：一年以内的短期国库券；
    - Treasury Notes：2年、3年、5年、7年、10年期国债；
    - Treasury Bonds：20年、30年期国债。
    Treasury calculate the yield rate following this process:
    国债现券市场报价
            ↓
    取得各只代表性国债的买方报价
            ↓
    根据债券价格计算收益率
            ↓
    拟合整条 Treasury par yield curve
            ↓
    读取1M、2Y、5Y、10Y、30Y等固定期限点
1.3 if you want to check the yield rate, search:
    期限	TradingView代码
    2年期收益率	TVC:US02Y
    5年期收益率	TVC:US05Y
    10年期收益率	TVC:US10Y
    30年期收益率	TVC:US30Y

    10年期收益率: 'https://www.tradingview.com/symbols/TVC-US10Y/'
1.4 How we use different data, and how we trade?
    - the relationships of **yield rate**, treasury bond price, and the **teasury bond future**
        收益率和期货价格两者通常反向运动：
        10Y收益率 US10Y 上升
                ↓
        10年国债价格下降
                ↓
        ZN期货价格通常下降
        
        所以在 TradingView 中可以对比：
        TVC:US10Y 上升
            ↓
        CBOT:ZN1! 下降

        trading_view 10年国债价格: 'https://www.tradingview.com/symbols/CBOT-ZN1%21/'
    - why would we use the interest rate first to analysis and then trade on the treasure bond future?
        因为两类数据承担不同作用：
        收益率数据
        → 描述和预测收益率曲线怎么变化
        国债期货
        → 把预测转化为实际可交易仓位
