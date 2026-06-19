# Pre-read: Time Series Analysis & Forecasting

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1</b><br/><i>(Python, Scikit-learn)</i><br/>Programming and ML fundamentals])
    C1[[Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(Scikit-learn, XGBoost)</i><br/>Supervised, unsupervised, and recommendation systems]]
  end

  CS{{Current Session<br/><b>Time Series Analysis &amp; Forecasting</b><br/><i>Seeing patterns across time</i><br/>Trend · Seasonality · ARIMA · Forecast evaluation}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Foundation for sequence modelling</b><br/>Time series skills directly lead into RNNs, LSTMs, and transformer-based forecasting in Module 3.])
    RV([Real-Life Value<br/><b>Forecast anything that changes over time</b><br/>Demand forecasting, stock prediction, energy load forecasting, and IoT anomaly detection all rely on these techniques.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3</b><br/><i>(PyTorch, HuggingFace)</i><br/>Deep learning, NLP, CV, and LLMs])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Forecast&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px
```

Your phone buzzes with an alert: the inventory management system is forecasting a stockout of a key component in three weeks. Production lines will slow, orders will pile up, and finance will want answers. The only thing standing between smooth operations and chaos is a single number — a forecast. And getting that number right depends on how well you can read the hidden patterns in historical data.

A simple average of past demand won't work — you would miss seasonal spikes and long-term drift. A straight line through the data ignores the weekly cycles and holiday surges that drive real business. If you were to treat time series data like any other machine learning problem — randomly shuffling rows and feeding them into a model — you would leak future information into your training set, making your predictions dangerously optimistic without ever realising it. The naive approach looks good on paper and fails in production.

Time series analysis demands a fundamentally different mindset: one that respects the order of events, extracts temporal features from the past, and evaluates forecasts without cheating by peeking at the future. That is where **Time Series Analysis and Forecasting** becomes essential.

What if you could look at any sequence of numbers — daily sales, hourly server traffic, monthly energy consumption — and instantly spot the long-term trend, the repeating seasonal rhythm, and the unusual anomalies that signal trouble? What if you could build a model that not only learns these patterns but also tells you how confident it is about next week's value, along with a reliable error margin? This session gives you the mental framework and the practical techniques to do exactly that — transforming a column of dates and numbers into a genuine competitive advantage.

Every time series has three hidden layers. The **trend** is the long-term direction — sales growing year over year, or gradually declining. The **seasonality** is the repeating cycle — higher retail sales every December, lower server traffic every weekend. The **noise** is the random variation that cannot be explained — a one-off spike from a viral post or a sudden dip due to a supplier delay. Think of it like listening to a song: the melody is the trend, the drumbeat is the seasonality, and the faint static in the background is the noise. Your job as a forecaster is to separate these three layers so you can project the melody and the beat forward while ignoring the static. In this session, you will explore **lag features** (yesterday's value as today's predictor), **rolling window statistics** (the moving average that smooths noise), and the **ARIMA** framework that brings autoregression, differencing, and moving averages into a single unified model. You will learn to read **ACF and PACF plots** — the diagnostic charts that tell you which past values matter — and you will evaluate your forecasts using **RMSE** and **MAPE**, the two metrics that separate a useful forecast from a misleading one.

In the **previous session**, you explored **Recommendation Systems** — building user-item matrices, applying collaborative filtering, and using matrix factorisation (SVD, ALS) to predict what a user might choose next. That work taught you to make predictions from historical interaction data, but it treated each interaction as an independent snapshot. Time series analysis takes the next logical step: it adds the dimension of time itself. Where recommendation systems ask "what does this user typically prefer across all past behaviour?", time series asks "what will happen next, given the order in which things happened before?" The same rigour around evaluation — precision@k for recommendations, RMSE and MAPE for forecasts — carries forward, but now you must also respect strict chronological order in your training data. The mental discipline you built around avoiding data leakage in ML pipelines becomes even more critical when the data itself marches forward in time.

In this pre-read, you will discover:

- How to recognise and separate **trend**, **seasonality**, and **noise** in any time series dataset.
- How to apply train/test splitting and feature engineering techniques that prevent **data leakage**.
- How to interpret **ACF** and **PACF** plots to select the right ARIMA model components.
- How to evaluate forecast accuracy using **RMSE** and **MAPE** and know when each metric matters.

---

## Why Standard Machine Learning Breaks on Time Series Data

Every machine learning course you have taken so far has hammered one habit: shuffle your data before splitting into train and test. Randomisation ensures your model sees a representative sample and generalises well. But the moment your data has a timestamp, shuffling becomes sabotage. Imagine training a sales prediction model on December data and testing it on November — your model would have already "seen" the holiday spike during training and would report unrealistically low error. This is the core trap of time series: **data leakage through time**.

A time series is defined by three components. The **trend** is the sustained upward or downward movement — think of a startup's monthly active users growing 15% quarter over quarter. The **seasonality** is the predictable pattern that repeats at fixed intervals — umbrella sales spiking every monsoon, or gym memberships surging every January. The **noise**, or residual, is everything left over: the random, unexplainable fluctuations that no model can (or should) capture. Your job is to decompose these three layers so you can project the trend and seasonality forward with confidence.

The fix is simple in principle and subtle in practice: split your data chronologically, not randomly. Train on the earliest 80% of your time period and test on the most recent 20%. Never let a future observation influence a past prediction. This rule extends to feature engineering — when you compute a rolling average or a lag feature, you must compute it using only data available at that point in time. Any feature that accidentally incorporates future information is a leak, and it will make your evaluation metrics lie to you.

## How Lag Features, Rolling Windows, and ARIMA Turn Time Into Predictors

Standard machine learning features describe the current state: price today, marketing spend this week, temperature this hour. But in time series, the most powerful predictor is often the past itself. A **lag feature** is simply the value from *N* periods ago — yesterday's sales to predict today's, or last month's energy bill to predict this month's. A **rolling window feature** aggregates a block of recent history — a 7-day moving average that smooths out the Monday-to-Sunday noise and reveals the underlying consumption pattern. Together, these two techniques transform a sequence into a supervised learning problem that any regression model can consume.

The ARIMA model formalises this intuition into a statistical framework. **AR** (autoregressive) means the current value depends on its own past lags — yesterday's temperature predicts today's. **I** (integrated) means differencing — subtracting the previous value from the current one — to remove trend and make the data stationary. **MA** (moving average) means the current value depends on past forecast errors, not just past values. Choosing the right number of lags and error terms is where **ACF** (autocorrelation function) and **PACF** (partial autocorrelation function) plots come in. The ACF plot shows how strongly a value correlates with its past at every lag; the PACF plot isolates the direct contribution of each lag, removing the influence of intermediate lags. Together, they act like a diagnostic scan — telling you whether your data is autoregressive (PACF cuts off sharply), moving-average (ACF cuts off sharply), or a mix of both.

Once your forecast is built, you need honest metrics. **RMSE** (Root Mean Squared Error) penalises large errors more heavily — perfect when a single disastrous forecast (like predicting zero demand for a high-value product) is far worse than several small ones. **MAPE** (Mean Absolute Percentage Error) expresses error as a percentage, making it interpretable across different scales — but it breaks down when actual values are close to zero. Using both gives you a complete picture: RMSE tells you about worst-case risk, and MAPE tells you about average relative accuracy.

## Where Time Series Forecasting Appears in Real Life

Retail and e-commerce rely on time series more than almost any other technique. Every major retailer forecasts demand at the SKU level — how many units of a specific shoe size will sell next week in each store — to optimise inventory, avoid stockouts, and reduce warehousing costs. A 5% improvement in forecast accuracy can translate into millions in working capital savings. Finance is another natural home: quantitative traders build ARIMA and related models to forecast stock volatility, currency exchange rates, and portfolio risk, while credit card companies use time series to detect unusual spending patterns that signal fraud. Energy utilities forecast load demand hour by hour, balancing electricity generation against consumption to prevent blackouts and minimise fuel costs — a task that becomes only more critical as renewable sources like solar and wind introduce greater variability into the grid. In healthcare, hospitals forecast patient admissions to staff emergency rooms and allocate beds; epidemiologists model case counts to anticipate outbreak surges. Even in manufacturing, sensor data from industrial equipment forms a continuous time series — predicting a bearing failure three days in advance through anomaly detection in vibration signals can prevent a production line shutdown that costs hundreds of thousands per hour. Every one of these domains starts with the same fundamental skill: decomposing a sequence of numbers into trend, seasonality, and noise, then projecting it forward with honest error bounds.

## What's Next

After this session, you will be able to:

- Decompose any time series into its trend, seasonality, and residual noise components using visual and statistical methods.
- Split time series data chronologically and engineer lag features and rolling window statistics without introducing data leakage.
- Interpret ACF and PACF plots to identify the autoregressive and moving-average orders for an ARIMA model.
- Build and evaluate ARIMA forecasts using RMSE and MAPE, and explain what each metric reveals about forecast quality.
- Recognise when a business problem demands a time-series-aware approach instead of standard supervised learning.

You do not need to build a production forecasting pipeline right now. The goal is to adopt a time-first mindset: **respect the order, and the patterns will reveal themselves.**

## Interesting Questions for the Live Session

- If you accidentally shuffled your time series before splitting into train and test, your RMSE might look unrealistically good — what kind of real-world business decisions could this false confidence lead a company to make?
- A 7-day rolling average smooths out daily fluctuations, but it also introduces a lag — in what business scenario would this lag be dangerous rather than helpful?
- ARIMA assumes the future statistically resembles the past — what happens to your forecast during a structural break like a pandemic, a policy change, or a competitor's sudden entry into the market?
- RMSE penalises large errors much more heavily than MAPE — when would you deliberately choose MAPE over RMSE, even if both seem to tell a similar story?

By the end of this session, time series should feel less like a special case and more like a practical superpower: **every business problem that evolves over time is yours to forecast.**
