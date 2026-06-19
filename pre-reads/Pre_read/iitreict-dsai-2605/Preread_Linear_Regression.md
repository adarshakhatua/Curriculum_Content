# Pre-read: Linear Regression

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Data Analysis with NumPy &amp; Pandas</b><br/><i>(NumPy, Pandas)</i><br/>Numeric computation and data wrangling])
    P2([Previous Module<br/><b>SQL for Data Science</b><br/><i>(SQL, SQLite)</i><br/>Relational databases and business queries])
    P3([Previous Module<br/><b>Statistics &amp; Data Visualization</b><br/><i>(Seaborn, Tableau)</i><br/>Statistical inference and BI dashboards])
    C1[[Current Module Until Previous Session<br/><b>Applied Machine Learning</b><br/><i>(Scikit-learn, ML concepts)</i><br/>ML lifecycle, supervised vs unsupervised]]
  end

  CS{{Current Session<br/><b>Linear Regression</b><br/><i>From describing data to predicting outcomes</i><br/>Simple · Multiple · Least Squares · MSE · R²}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Foundation for all supervised learning</b><br/>Linear Regression builds the mental model of learning from labeled data that every future ML algorithm extends and refines.])
    RV([Real-Life Value<br/><b>Everyday prediction in industry</b><br/>From real estate pricing and sales forecasting to financial risk modeling, linear regression is the first tool analysts reach for when they need to predict a continuous number.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>GenAI for Data Science</b><br/><i>(LLMs, RAG)</i><br/>Large language models and retrieval-augmented generation])
    F2([Upcoming Module<br/><b>Capstone Project</b><br/><i>(Full stack ML, Streamlit)</i><br/>End-to-end solution design and delivery])
  end

  P1 ==>|&nbsp;Data Wrangling&nbsp;| P2
  P2 ==>|&nbsp;Statistical Thinking&nbsp;| P3
  P3 ==>|&nbsp;ML Foundation&nbsp;| C1
  C1 ==>|&nbsp;Continuous Prediction&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1
  F1 ==>|&nbsp;Deliver&nbsp;| F2

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2,P3,C1 previous;
  class CS current;
  class CV,RV value;
  class F1,F2 future;
  linkStyle default stroke-width:3px
```

You are handed a dataset of 10,000 houses sold in your city last year. Each row has the square footage, number of bedrooms, location score, and the final sale price. A colleague asks: "If I find a 1,500 sq ft, 3-bedroom house in a neighbourhood with an 8/10 location score, what should I offer?" Your instinct says there is a way to calculate it — some kind of rule that turns features into a price.

The naive approach is to take averages — the average price per square foot, the average premium for an extra bedroom. But these averages fight each other. A bigger house usually has more bedrooms, so pulling one number without controlling for the others gives you a distorted answer. Worse, you cannot tell how reliable your estimate is. You might be off by ₹50 lakhs or ₹5 lakhs, and you have no way to know.

What you need is a single mathematical formula that weighs each feature appropriately, finds the best-fitting line through your data, and gives you both a prediction and a measure of how good that prediction is. That is where **Linear Regression** becomes essential.

What if you could open any spreadsheet with historical data — sales figures, customer churn rates, inventory levels — and build a model in under 30 seconds that predicts the next quarter's numbers with measurable accuracy? Not a guess. Not a gut feel. A number with a documented uncertainty margin. And then explain to your manager exactly which factors drive the prediction and by how much. That capability starts here. By the end of this session, you will write Python code that learns from data automatically and produces predictions on command.

At its core, **Linear Regression** is a statistical method that models the relationship between one or more independent variables (features) and a continuous dependent variable (target). In its simplest form — **Simple Linear Regression** — this is a straight line: $y = mx + c$. With multiple features, it becomes a hyperplane: $y = w_1x_1 + w_2x_2 + ... + b$. Think of it like trying to balance a plank on a fulcrum. Each data point pulls the plank down. The **Least Squares** method finds the position of the plank (the line) where the total downward pull (the sum of squared errors) is smallest. The **coefficients** tell you how hard each feature pulls, and the intercept is where the plank would sit if all features were zero. In this session, you will explore **Simple vs Multiple regression**, understand how **coefficients** and **intercepts** define the prediction line, learn how **Least Squares** finds the optimal fit, and use **MSE** and **R²** to evaluate how trustworthy your predictions are.

In the **previous session**, you were introduced to the machine learning lifecycle — the end-to-end process of framing a problem, preparing data, training a model, evaluating it, and deploying it. You learned the critical distinction between supervised and unsupervised learning and got your hands on scikit-learn for the first time. That session gave you the conceptual scaffolding of ML: what it is, when to use it, and the vocabulary to talk about it. This session turns that scaffolding into your first working model. Where session 17.1 showed you the map of ML, session 17.2 teaches you how to drive. Linear Regression is the simplest form of supervised learning — and because it is simple, every concept you learn here (training a model on historical data, interpreting coefficients, evaluating predictions using metrics) transfers directly to every more complex algorithm you will encounter later, from Logistic Regression to Neural Networks.

In this pre-read, you will discover:

- How to **build** a model that predicts a continuous value from one or more input features using the linear regression framework
- How to **interpret** regression coefficients and the intercept to explain what drives predictions
- How to **apply** the Least Squares method to find the best-fitting line through your data
- How to **evaluate** model performance using MSE and R² to know whether your predictions are trustworthy

---

## Why a Straight Line Can Be a Powerful Prediction Tool

A straight line is the simplest possible relationship between two variables: as one goes up, the other goes up (or down) at a constant rate. When you fit a **Simple Linear Regression**, you are asserting that the relationship between your feature and your target is, to a first approximation, linear. That turns out to be surprisingly useful. In real estate, adding 100 square feet to a house tends to add a roughly constant amount to the price. In retail, spending an extra ₹10,000 on advertising tends to produce a predictable increase in sales. Not perfectly constant, but close enough that a straight line captures most of the signal.

The **Least Squares** method is what makes this rigorous. Instead of eyeballing a line, it calculates the vertical distance from every data point to your line (the **residual**), squares it (so positive and negative errors do not cancel), and finds the line that minimises the sum of all those squared distances. You get two numbers: the **slope** (how much the target changes for a one-unit increase in the feature) and the **intercept** (the predicted value when the feature is zero). These are your learned parameters. With them, you can predict any new input instantly.

What makes a straight line deceptively powerful is that it forces you to think causally. The slope is not just a number — it is a statement about the world: "one additional bedroom adds this much to the price, all else being equal." That interpretability is rare in more complex models. It is also a sharp check against overconfidence. If the data points scatter widely around your line, the line is honest about it — your **Mean Squared Error (MSE)** will be large, and that is valuable information in itself.

## When One Feature Is Not Enough — The Leap to Multiple Regression

House prices do not depend on square footage alone. Location, number of bedrooms, age of the property, proximity to schools, and market conditions all play a role. **Multiple Linear Regression** extends the straight line to a hyperplane — a linear combination of all your features — and learns a coefficient for each one. The interpretation changes slightly: every coefficient now represents the expected change in the target when that feature increases by one unit, **holding all other features constant**.

This "holding all others constant" is the key insight. In simple regression, the slope for square footage captures both the effect of size and the fact that bigger houses tend to have more bedrooms. In multiple regression, the coefficient for square footage isolates the pure effect of size after removing the influence of bedrooms. This is why multiple regression is the workhorse of observational studies across economics, epidemiology, and social science — it lets you ask "what is the effect of X on Y, controlling for Z?" without running a controlled experiment.

But multiple regression introduces a new challenge: **R²**, the proportion of variance explained, always increases when you add more features, even if those features are random noise. That is why you cannot trust R² alone. A high R² might mean your model is genuinely accurate, or it might mean you have overfit by throwing in too many variables. You need to pair it with MSE (which penalises large errors in the original units of your target) and with your own judgment about whether the coefficients make intuitive sense. A negative coefficient for number of bedrooms in a housing model should make you stop and investigate — it might reveal multicollinearity, data leakage, or a genuine market anomaly worth exploring.

## Where Linear Regression Appears in Real Life

Linear regression is not a classroom exercise — it is embedded in the daily workflow of analysts, economists, and engineers across nearly every industry. In **real estate**, automated valuation models (AVMs) used by Zillow, Redfin, and property tax assessors are often built on multiple regression, predicting sale prices from features like square footage, lot size, bedrooms, and location scores. In **finance**, credit risk teams use logistic regression (a close cousin) to estimate default probability, but linear regression itself appears in value-at-risk models, insurance premium pricing, and portfolio return forecasting. In **retail and e-commerce**, demand forecasting systems predict how many units of a product will sell next week based on price, promotions, seasonality, and competitor pricing — all grounded in the same least-squares framework. In **healthcare**, researchers use linear regression to model patient outcomes: for example, the relationship between a drug dosage and blood pressure reduction, controlling for age, weight, and baseline health. In **economics**, it is the tool of choice for estimating GDP growth, unemployment elasticity, and the impact of policy changes on economic indicators, often feeding directly into government budget planning and central bank decisions. The common thread across all these domains is the same: when you need to predict a continuous number and explain what drives it, linear regression is the first model you reach for.

## What's Next

After this session, you will be able to:

- Train a linear regression model using scikit-learn on a real dataset with multiple features
- Interpret regression coefficients and the intercept to explain each feature's contribution to the prediction
- Evaluate model fit using Mean Squared Error and R-squared metrics
- Recognise when a linear model is appropriate and when its assumptions break down
- Apply the distinction between simple and multiple regression to choose the right approach for your data

You do not need to memorise any derivations or prove any theorems right now. The goal is a straight line, learned from data, that turns raw numbers into actionable predictions: **linear regression is the gateway to every supervised learning model you will ever build.**

## Interesting Questions for the Live Session

- If your R² is 0.95, does that mean your model is always accurate enough for production? What could still go wrong?
- When you add more features to a regression, R² can never decrease — so why would you ever stop adding features?
- The Least Squares method squares the errors before summing them. What changes if you used absolute errors instead, and when might that be better?
- Your model predicts house prices, but one coefficient is negative when you intuitively expect it to be positive. Should you trust the model or your intuition?

By the end of this session, linear regression should feel less like a formula from a textbook and more like a practical reasoning tool: **a way to let data speak in a straight line.**
