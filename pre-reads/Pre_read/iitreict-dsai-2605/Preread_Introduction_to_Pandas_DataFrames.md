# Pre-read: Introduction to Pandas DataFrames

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1: Programming &amp; Data Foundations (Weeks 1-4)</b><br/><i>(Python, APIs)</i><br/>Python programming and API data fetching])
    C1[[Current Module Until Previous Session<br/><b>Module 2: Data Analysis with NumPy &amp; Pandas (Weeks 5-8)</b><br/><i>(NumPy, Vectorization)</i><br/>NumPy arrays, vectorized ops, linear algebra]]
  end

  CS{{Current Session<br/><b>Introduction to Pandas DataFrames</b><br/><i>From numbers to tables — the data scientist's workspace</i><br/>Pandas Core · Series vs DataFrames · Read CSV/Excel · Basic Inspection}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Foundation for all data analysis</b><br/>Pandas DataFrames power every subsequent module — SQL, statistics, ML, and GenAI all rely on DataFrame skills.])
    RV([Real-Life Value<br/><b>The industry standard for data work</b><br/>Pandas is the most-used Python library in data science — reading CSV/Excel into DataFrames is the first skill employers expect.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3: SQL for Data Science (Weeks 9-12)</b><br/><i>(SQL, Databases)</i><br/>Querying and combining datasets])
    F2([Upcoming Module<br/><b>Module 4: Statistics &amp; Data Visualization (Weeks 13-16)</b><br/><i>(Statistics, Visualization)</i><br/>Describing and visualizing data])
    F3([Upcoming Module<br/><b>Modules 5 &amp; 6: Applied ML &amp; GenAI (Weeks 17-24)</b><br/><i>(Scikit-learn, LLMs)</i><br/>Predictive models and AI applications])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;DataFrames&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1
  F1 ==>|&nbsp;Analyze&nbsp;| F2
  F2 ==>|&nbsp;Predict&nbsp;| F3

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,C1 previous;
  class CS current;
  class CV,RV value;
  class F1,F2,F3 future;
  linkStyle default stroke-width:3px
```

You have just been handed a folder with fifteen CSV files — sales records from every region your company operates in. Each file contains thousands of rows with columns like `transaction_id`, `amount`, `region`, and `date`, and your manager wants to know which region grew fastest last quarter. The data is all there, but it is scattered across files, inconsistently formatted, and mixed with missing values and stray text.

Your first instinct might be to open each file in a spreadsheet and start scanning. But one file is missing the header row, another has dates in `MM/DD/YYYY` while a third uses `DD-MM-YY`, and several cells in the `amount` column contain text like "N/A" or a dash. Cleaning this manually would take hours, and you would still need to combine everything into a single view before you could compare regions side by side. The raw numbers are there, but extracting meaning from them feels like searching for a signal in static.

This is exactly where a tool built for structured data analysis becomes essential. **Pandas** is a Python library designed to load, clean, explore, and transform tabular data with just a few lines of code. Instead of wrestling with individual cells, you declare what you want and Pandas handles the heavy lifting. That is where **Pandas DataFrames** become essential.

What if you had to analyse a year's worth of customer transactions — millions of rows scattered across a dozen CSV files — and produce a single-page summary of buying patterns by region, category, and season? Doing this cell by cell in a spreadsheet would take days, and the moment you introduced a new data source, you would have to start over. But once you know how to load data into a **DataFrame** and inspect its structure with a handful of commands, you can ask complex questions of your data in seconds. This session is the first step toward that capability: taking raw files and turning them into a workspace you can actually query.

When you work with data in Python, you typically start with raw structures — lists of lists, dictionaries, or NumPy arrays. These are powerful, but they lack context: a NumPy array tells you the numbers, but not which column is "revenue" and which is "date". A **DataFrame** solves this by wrapping data in labeled rows and columns, much like a spreadsheet, but programmable. A **Series** is a single labeled column — think of it as one column of a DataFrame, complete with its own name and data type. Together, the DataFrame and Series form the two core data structures of Pandas.

Think of NumPy as a filing cabinet full of unlabeled drawers (raw arrays) and Pandas as the same cabinet with every drawer clearly labelled and inventoried. You can still access the numbers directly, but you can also ask for everything in the "revenue" drawer and get it back instantly, without remembering which column index it was stored in. In this session, you will explore the **core operations** that make Pandas tick: how a **Series** differs from a **DataFrame**, how to **read CSV and Excel files** into these structures, and how to perform **basic inspection** — checking the shape of your data, the column names, data types, and the first few rows — before diving deeper into analysis.

In the **previous session**, you reshaped NumPy arrays, transposed matrices, and generated synthetic data for experiments. You learned to think of data as multi-dimensional blocks of numbers that could be manipulated with high-performance vectorized operations, all without writing a single loop. Now consider this: every column of a Pandas DataFrame is backed by a NumPy array under the hood. When you load a CSV into a DataFrame, Pandas internally stores each column as a NumPy array and overlays it with labels, column names, and data-type awareness. The reshaping, transposing, and indexing skills you already built with NumPy are the engine that powers Pandas — this session adds the steering wheel, the dashboard, and the road map.

In this pre-read, you will discover:
- How to **load** a CSV or Excel file into a DataFrame using `pd.read_csv()` and `pd.read_excel()`
- How to **distinguish** between a Series and a DataFrame and understand when to use each
- How to **inspect** a new dataset using `.head()`, `.info()`, `.describe()`, and `.shape`
- How to **recognise** the core operations that make Pandas the standard tool for tabular data in Python

---

## What Makes a DataFrame Different From a NumPy Array

At first glance, a DataFrame and a 2D NumPy array look similar: both store data in rows and columns, and both support fast mathematical operations. But the similarity ends there. A NumPy array requires every element to be the same data type — if you store integers, every value must be an integer. A DataFrame, by contrast, allows each column to have its own type. One column can hold integers, another floats, another strings, and another dates — all in the same table. This matters because real-world data almost never arrives in a single uniform type. Your sales dataset has numeric amounts, text categories, and date columns mixed together, and a DataFrame handles this natively.

The second difference is labelling. In a NumPy array you access values by integer index: `arr[2, 5]` gives you the element at row 2, column 5. In a DataFrame you can still do that, but you can also access by column name: `df["amount"]` returns the entire "amount" column, and `df.loc[2, "amount"]` returns the specific value at row 2 of that column. This shift from positional to label-based access is not a minor convenience — it makes your code self-documenting and far less error-prone. When you read `df["amount"].mean()`, you know exactly what is being averaged, without having to remember that "amount" was column index 3. A **Series** is the bridge between these two worlds: it is a single column of data with an index, backed by a NumPy array, but with labels that give every value a name and a position.

## What Happens When You Read a CSV Into a DataFrame

When you call `pd.read_csv("sales.csv")`, Pandas performs a series of operations before you ever see the data. It reads the file line by line, identifies the header row (by default the first row), and uses those values as column names. It then scans each column to infer the most likely data type — integers become `int64`, decimals become `float64`, text becomes `object`, and date-like strings are converted to `datetime` if detected. This type inference is both powerful and fallible: a column that looks like numbers but contains a single "N/A" string will be downgraded to `object` type, and a date column in an unusual format may remain as plain text unless you tell Pandas how to parse it.

This is where **basic inspection** becomes your first real skill. After loading any file, you should immediately run `df.head()` to see the first five rows and confirm the columns look correct. Then `df.info()` reveals the data type of every column, the number of non-null values, and the memory usage — a quick health check for your dataset. `df.describe()` gives you summary statistics (count, mean, min, max) for every numeric column, letting you spot outliers or impossible values at a glance. And `df.shape` tells you the dimensions of your table in (rows, columns), which is essential when you start merging or filtering data later. These four methods form the foundation of every Pandas workflow, and you will use them in virtually every session that follows.

## Where DataFrames Appear in Real Life

The DataFrame is not a classroom abstraction — it is the primary data structure used across the data industry. In **finance**, analysts load years of stock tick data into DataFrames to calculate moving averages, detect volatility patterns, and backtest trading strategies. A single `pd.read_csv()` call can pull in a decade of daily prices, and a chain of Pandas operations turns raw ticks into a risk report. In **e-commerce**, DataFrames hold product catalogs, customer profiles, and transaction logs; a data scientist might load sales data, join it with customer demographics, and compute customer lifetime value for every segment — all without leaving the Pandas ecosystem.

In **healthcare**, patient records, lab results, and medication schedules arrive as CSV exports from hospital systems. A DataFrame allows a researcher to filter for a specific diagnosis, group patients by age range, and compute average recovery times across treatment groups, all while keeping every row linked to its original patient ID. In **marketing**, campaign performance data from platforms like Google Ads and Facebook are downloaded as CSV reports and loaded into DataFrames for A/B test analysis, cost-per-acquisition calculations, and cohort retention studies. And in **operations** and supply chain, inventory levels, shipping logs, and warehouse throughput data live in DataFrames, where they are inspected for anomalies and summarised into executive dashboards. Across every one of these domains, the workflow begins the same way: read the file, inspect the structure, and start asking questions. That is the skill this session builds.

## What's Next

After this session, you will be able to:

- Load a CSV file into a DataFrame using `pd.read_csv()` and confirm its structure with `.head()` and `.info()`
- Distinguish between a **Series** and a **DataFrame** and choose the right structure for a given data task
- Access specific columns, rows, and individual cells using bracket notation and the `.loc` accessor
- Read an Excel worksheet into a DataFrame using `pd.read_excel()` with the target sheet specified by name
- Use `.describe()`, `.shape`, and `.dtypes` to perform a rapid initial assessment of any dataset's size, distribution, and column types

You do not need to memorise every Pandas method right now. The goal is to see tabular data not as a grid of numbers but as a structure you can query and trust — one cell, column, and condition at a time.

## Interesting Questions for the Live Session

- What happens when you read a CSV with inconsistent column counts across rows — does Pandas fail silently or raise an error, and how would you detect the problem after loading?
- If a DataFrame and a NumPy array both hold the same 2D numeric data, what operations become possible with the DataFrame that are cumbersome or impossible with the array alone?
- A Series shares many method names with a DataFrame (like `.mean()` and `.dropna()`) — is a Series just a single-column DataFrame, or is there a deeper structural difference that affects how you write your code?
- When you call `pd.read_csv()` on a 2 GB file, what actually happens under the hood — does Pandas load the entire file into memory at once, and what options does it offer if the file does not fit?

By the end of this session, DataFrames should feel less like a fancy spreadsheet and more like a programmable workspace where your data becomes queryable: **tables you can talk to.**
