# Pre-read: Introduction to Window Functions

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1: Programming &amp; Data Foundations</b><br/><i>(Python, APIs)</i><br/>Built core programming and data skills])
    P2([Previous Module<br/><b>Module 2: Data Analysis with NumPy &amp; Pandas</b><br/><i>(NumPy, Pandas)</i><br/>Mastered structured data analysis])
    C1[[Current Module Until Previous Session<br/><b>Module 3: SQL for Data Science</b><br/><i>(SQL Joins, GROUP BY)</i><br/>DB schema, joins, aggregations with GROUP BY and HAVING]]
  end

  CS({{Current Session<br/><b>Introduction to Window Functions</b><br/><i>A new dimension of row-wise calculation</i><br/>OVER clause · ROW_NUMBER · RANK · DENSE_RANK · Running totals}})

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Powers advanced analytics ahead</b><br/>Enables statistical comparisons, feature engineering, and sequential analysis across future modules.])
    RV([Real-Life Value<br/><b>Industry-standard data analysis</b><br/>Used daily for ranking sales, computing running totals, and detecting anomalies in time-series data.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 4: Statistics &amp; Data Visualization</b><br/><i>(Hypothesis Testing, BI Tools)</i><br/>Statistical reasoning and visual storytelling])
    F2([Upcoming Module<br/><b>Module 5: Applied Machine Learning</b><br/><i>(Regression, Classification)</i><br/>Building predictive models with ML])
    F3([Upcoming Module<br/><b>Module 6: GenAI for Data Science</b><br/><i>(LLMs, RAG)</i><br/>Modern GenAI in data science])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| C1
  C1 ==>|&nbsp;Window&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1
  F1 ==>|&nbsp;Analyze&nbsp;| F2
  F2 ==>|&nbsp;Predict&nbsp;| F3

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2,C1 previous;
  class CS current;
  class CV,RV value;
  class F1,F2,F3 future;
  linkStyle default stroke-width:3px
```

Imagine you are analysing monthly sales data for a retail chain with 50 stores. Your manager asks a straightforward question: "For each store, show me the total sales so far, month by month, and rank each store within its region." You pull up your SQL query — and pause. A simple GROUP BY can give you totals per store, but it collapses every row into one summary. You need each row preserved, with an additional column that shows the running total. The query you have been writing so far cannot do both at the same time.

The intuitive approach might be to write a subquery, or join the table to itself, or pull the data into Python and loop through it. But subqueries become slow and tangled when you need multiple rankings. Self-joins can balloon into massive temporary tables. And exporting to Python defeats the purpose of keeping your analysis in the database where the data lives. The harder the problem gets, the more you realise that SQL as you know it has a gap — a gap between summarising and preserving detail together on the same row.

That gap is exactly what **window functions** are built to fill. They let you perform calculations across a set of rows while keeping each row intact — giving you running totals, rankings, and moving averages without collapsing or duplicating your data. That is where **window functions** become essential.

What if you could write a single SQL query that ranks every product in your inventory by sales performance, computes a running total of revenue per quarter, and identifies the top three customers in each region — all without subqueries, self-joins, or post-processing in Python? What if you could compare each row's value against the previous row in a single line of SQL? This session gives you exactly that power: the ability to perform row-wise calculations across partitions of your data with the elegance and speed that only window functions can provide.

In SQL, every query you have written so far either returned raw rows or summarised them into groups. A **window function** does something different: it performs a calculation across a set of rows — called a **window** — while still returning each individual row. Think of it as opening a temporary window over a portion of your table, computing something, and then moving to the next portion. The **OVER clause** is what defines that window: it tells SQL which rows to look at for each calculation. If GROUP BY is a sledgehammer that collapses rows, a window function is a scalpel that adds context without removing detail. In this session, you will explore four foundational techniques: **ROW_NUMBER** for assigning unique row positions, **RANK** and **DENSE_RANK** for handling ties in rankings, and **running totals** for cumulative computations. Together, these tools form the backbone of almost every advanced SQL analysis you will encounter.

In the **previous session**, you mastered advanced aggregations with **GROUP BY** and **HAVING** — learning how to group rows, filter groups, and use `COUNT(DISTINCT)` and `CASE WHEN` to build sophisticated summary tables. That session gave you the power to collapse data into meaningful summaries. But what if you need both the detail and the summary on the same row? That is where GROUP BY reaches its limit and window functions take over. The grouping mental model you built — partitioning data into logical buckets — is the exact foundation that the OVER clause extends, except window functions let you retain every row while still computing across those partitions. You are not starting from scratch; you are taking one more step along the same path.

In this pre-read, you will discover:

- How to **apply** the OVER clause to define calculation windows across table rows.
- How to **build** row numbers and rankings using ROW_NUMBER, RANK, and DENSE_RANK.
- How to **recognise** the difference between RANK and DENSE_RANK when handling tied values.
- How to **connect** running totals to ordered data within a window frame.

---

## How the OVER Clause Unlocks Row-Wise Calculations

The core difference between a window function and a regular aggregate function lies in the **OVER clause**. When you write `SUM(sales)`, SQL normally expects a GROUP BY to tell it how to collapse rows. But when you write `SUM(sales) OVER (PARTITION BY region ORDER BY month)`, SQL understands: do not collapse the rows. Instead, for each row, compute the sum of sales within its region, ordered by month, and place that result right next to the original value.

Think of OVER as defining a sub-frame. The **PARTITION BY** sub-clause splits your table into logical groups — similar to GROUP BY — but without merging them. Every row stays in the result. The **ORDER BY** sub-clause inside OVER determines the sequence in which the calculation is applied. For running totals, order matters: each row's cumulative sum includes all rows before it within the same partition. Leave out ORDER BY and the window function treats all rows in the partition as equal, which changes the result entirely.

This is not just a syntax detail. It represents a shift in how you think about computation: from "collapse-first, ask-later" to "compute-alongside, preserve-everything." Mastering the OVER clause is what separates intermediate SQL users from advanced ones. Once you internalise this pattern, you will see opportunities to use it in almost every query you write.

## RANK vs. DENSE_RANK: What Happens When Values Are Tied?

When you rank items, ties create a dilemma. Suppose three students each score 95 on an exam. If you assign them ranks 1, 2, and 3, you are implying one is better than the other even though they scored the same. If you assign them all rank 1, you need a rule for what comes next. That is exactly the problem **RANK** and **DENSE_RANK** solve differently.

**RANK** assigns the same number to tied rows, then skips the next position. If two students tie for first, RANK gives them both 1, then skips 2 — the next student gets rank 3. **DENSE_RANK** also assigns the same number to tied rows, but never skips positions — so the student after the tie gets rank 2. The choice between them depends on whether you want a "gapped" or "gapless" ranking. Sports leaderboards typically use RANK because the gap signals the margin of victory, while dense numbering is more common in reporting where you want consecutive row numbers.

This distinction matters whenever you build ranked reports, because picking the wrong function can misrepresent the data. A marketing team that sees rank gaps might assume a significant performance difference when a tie in score is the only issue. Knowing when to use each function is a hallmark of an analyst who thinks carefully about how their tools frame the story.

## Where Window Functions Appear in Real Life

In **retail and e-commerce**, window functions power the "top sellers this month" dashboards, compute running revenue totals by region, and rank products by customer ratings — all without leaving the database. In **finance and banking**, analysts use window functions to calculate moving averages of stock prices, compare quarter-over-quarter loan growth, and detect transactions that deviate from a customer's typical spending pattern. A running total of daily transactions, for example, instantly reveals whether the month is trending above or below forecast.

In **healthcare**, hospital systems use window functions to rank departments by patient volume, compute cumulative patient admissions during flu season, and compare readmission rates across consecutive time periods. In **logistics and supply chain**, running totals over shipment volumes help forecast warehouse capacity, while RANK is used to identify the top-performing delivery routes and flag underperforming ones. Every time you need to compare a row to its neighbours, or compute a cumulative measure without losing detail, window functions are the tool you will reach for. They are a standard part of the SQL toolkit in every major database — PostgreSQL, MySQL, Snowflake, BigQuery — which means the skill transfers directly across platforms.

## What's Next

After this session, you will be able to:

- Write analytic queries using the OVER clause to define calculation windows.
- Assign unique row numbers within partitions using ROW_NUMBER.
- Rank data with RANK and DENSE_RANK, choosing the correct function for tied values.
- Compute running totals and cumulative sums over ordered data.
- Combine window functions with WHERE and ORDER BY clauses for clean, efficient reports.

You do not need to memorise every window function variant right now — the OVER clause is the key that unlocks them all. The goal is to see window functions not as an advanced SQL feature, but as the natural next step after GROUP BY: a way to preserve detail while gaining context.

## Interesting Questions for the Live Session

- What happens to a window function if you omit the ORDER BY inside the OVER clause — does the result change, and why?
- When would you deliberately choose RANK over DENSE_RANK in a business report, and how might a stakeholder misinterpret the gap?
- Can you use a window function inside a WHERE or GROUP BY clause, and what does that tell you about the order of query execution?
- If a running total query returns unexpected results, how would you debug whether the PARTITION BY or ORDER BY sub-clause is causing the issue?

By the end of this session, window functions should feel less like an obscure SQL feature and more like a practical extension of the grouping logic you already know: **a window is just a group that keeps its rows.**
