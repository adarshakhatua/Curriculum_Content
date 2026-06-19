# Pre-read: SQL + Python Integration

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1(["Previous Module<br/><b>Module 1: Programming &amp; Data Foundations (Weeks 1-4)</b><br/><i>(Python, APIs)</i><br/>Core Python programming and data fetching"])
    P2(["Previous Module<br/><b>Module 2: Data Analysis with NumPy &amp; Pandas (Weeks 5-8)</b><br/><i>(NumPy, Pandas)</i><br/>Numeric computing and structured data manipulation"])
    C1[["Current Module Until Previous Session<br/><b>Module 3: SQL for Data Science (Weeks 9-12)</b><br/><i>(SQL, Relational Databases)</i><br/>DB schema, joins, aggregations, window functions, time-series SQL"]]
  end

  CS{{"Current Session<br/><b>SQL + Python Integration</b><br/><i>From query tool to programmable pipeline</i><br/>SQLite · Postgres · SQLAlchemy · Connection strings · Cursors"}}

  subgraph Value["Why This Matters"]
    CV(["Course Value<br/><b>Bridging SQL to ML pipelines</b><br/>By learning to query databases directly from Python, you eliminate the manual export-import step, enabling the real-time data flows that power every subsequent module — from statistical analysis in M4 to model training in M5 and LLM-based retrieval in M6."])
    RV(["Real-Life Value<br/><b>From silos to live data pipelines</b><br/>In any data-driven organisation, data lives in databases, not CSV files — this session equips you with the skill to write production scripts that extract, transform, and load (ETL) data directly from a company's operational database into your analysis or ML workflow."])
  end

  subgraph Future["Where This Leads Next"]
    F1(["Upcoming Module<br/><b>Module 4: Statistics &amp; Data Visualization (Weeks 13-16)</b><br/><i>(Statistics, Tableau/Seaborn)</i><br/>Statistical reasoning and visual storytelling"])
    F2(["Upcoming Module<br/><b>Module 5: Applied Machine Learning (Weeks 17-20)</b><br/><i>(Scikit-learn, ML Models)</i><br/>Predictive modeling and evaluation"])
    F3(["Upcoming Module<br/><b>Module 6: GenAI for Data Science (Weeks 21-24)</b><br/><i>(LLMs, Vector DBs)</i><br/>Generative AI and semantic search"])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| C1
  C1 ==>|&nbsp;Bridge&nbsp;| CS
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

You have just finished writing a complex SQL query — a multi-step CTE that computes 90-day user retention, churn probability, and revenue per cohort. The output is a clean, four-column table sitting inside your PostgreSQL database. Now you need to feed that data into a Python script to calculate a logistic regression model. You export the results to CSV, open it in a notebook, re-import it, and immediately notice the datetime columns have lost their type, the numeric precision has shifted, and there are stray quote marks from the CSV formatting.

Every data scientist has been here. The manual export-import dance is not just annoying — it is brittle, error-prone, and completely unsuited for any workflow that needs to run on a schedule or handle fresh data daily. When your database contains millions of rows, CSV exports become storage nightmares. When your stakeholders expect a live dashboard, a static file is useless. The tension is clear: your data lives in a database, but your analysis tools live in Python, and the bridge between them is held together by guesswork.

The solution is not to abandon either side — it is to make Python speak SQL natively, turning your database into a first-class data source that your code can query, manipulate, and write back to in real time. That is where **SQL + Python Integration** becomes essential.

What if you could write a Python script that connects to your company's production database, runs a complex analytical query, loads the result directly into a Pandas DataFrame, performs feature engineering, trains an ML model, and writes the predictions back — all without a single CSV file touching your disk? What if your entire data pipeline, from raw database rows to deployed model, fit inside a single, auditable, version-controlled script? This is not theoretical. This is the standard workflow of every data engineer and ML engineer in the industry. And this session gives you the key to unlock it.

The core concept is simple: a database and a Python program speak different languages. A database understands SQL — tables, rows, and relational operations. Python understands objects, functions, and DataFrames. To make them communicate, you need a **connector** — a piece of software that translates between the two worlds. The most common connectors are database-specific libraries such as **sqlite3** (for SQLite databases) and **psycopg2** (for PostgreSQL), which implement a standard protocol called **DB-API 2.0**. This protocol defines a shared vocabulary of **connection strings**, **cursors**, and execution methods that every Python database library follows.

Think of it like ordering food in a foreign country. You know what you want (the data), and the chef knows how to prepare it (the database). But neither of you speaks the other's language. A connector is your multilingual waiter: you tell the waiter what you need in Python, the waiter translates it to SQL for the database, brings back the results, and hands them to you as a Python object you can work with. The **connection string** is the restaurant's address — it tells Python exactly where the database lives, what credentials to use, and which protocol to speak. The **cursor** is your personal waiter assigned to your table — it carries each query you make and brings back the response.

In this session, you will explore two sides of the integration story. First, you will work hands-on with **SQLite** (a lightweight, file-based database perfect for prototyping and local work) and **PostgreSQL** (a full-featured production database). Then you will be introduced to **SQLAlchemy**, an abstraction layer that lets you write the same Python code regardless of which database engine sits underneath — a skill that becomes indispensable when you move between projects, teams, or companies that use different database systems.

In the **previous session** (11.3 — Lead, Lag & Time-Series SQL), you learned to analyse data changes over time entirely within SQL — comparing current rows with previous ones, calculating growth rates, and computing moving averages directly in the database. Those skills turned SQL from a simple retrieval tool into a computational engine capable of sophisticated time-series analysis. But every time-series insight you produced in that session was trapped inside the database console. You could see it, admire it, and even export it — but you could not bring it into a Jupyter notebook to visualise with Seaborn, combine it with external data from an API, or feed it into a machine learning model. This session hands you the missing piece: the ability to execute those same powerful queries from within Python and receive the results as first-class Python objects that you can manipulate, visualise, and pass to any library in the data science ecosystem.

In this pre-read, you will discover:
- How to connect to SQLite and PostgreSQL databases from Python using connection strings
- How to execute SQL queries through a cursor and retrieve results as structured data
- How to use SQLAlchemy to write database-agnostic Python code
- How to bridge the gap between SQL results and Pandas DataFrames for analysis

---

## Why Connection Strings Are More Than Just Addresses

A connection string looks deceptively simple — a single line of text that bundles together a database driver, a server address, a port number, a username, a password, and a database name. Something like `postgresql://user:pass@localhost:5432/mydb`. It is easy to copy-paste it from a configuration file and move on. But every piece of that string encodes a critical decision that affects security, performance, and reliability.

Consider the **driver** segment at the start. `sqlite:///data.db` and `postgresql://user:pass@host/db` look similar, but they trigger completely different connection behaviours. SQLite creates or opens a local file — no server, no authentication, no network overhead. PostgreSQL opens a TCP socket to a remote server, negotiates a handshake, authenticates your credentials, and allocates a backend process on the database server. Choosing the wrong driver means your code will fail before it runs a single query. The **port** and **host** fields determine whether you are talking to a local development database on `localhost` or a cloud-hosted production instance on an AWS endpoint — and a single wrong character here can route sensitive queries to the wrong environment.

The deeper skill is learning to treat connection strings as configuration, not code. Hardcoding a connection string with plain-text credentials inside a script is one of the most common security mistakes in data science code. By the end of this session, you will understand not just how to construct a connection string, but how to manage it safely — using environment variables, `.env` files, or secret management tools — so that your database credentials never end up in a GitHub repository.

## How Cursors Turn Python Into a Database Client

When you connect to a database, you have not yet done anything with it. The connection is like opening a phone line to a server — the line is open, but nobody is talking. To actually communicate, you create a **cursor**. A cursor is a control structure that lets you send SQL statements to the database one at a time and retrieve their results incrementally. It is the workhorse of every database interaction in Python.

The typical pattern has three steps: create a cursor, execute a query, and fetch the results. `cursor.execute("SELECT * FROM users")` sends the SQL string to the database engine, which parses it, optimises a query plan, and runs it against the data. The results stay on the database server — they are not yet in your Python program. Only when you call `cursor.fetchone()`, `cursor.fetchmany(100)`, or `cursor.fetchall()` does the data cross the network and materialise as Python objects. This separation is deliberate: it lets you handle large result sets without loading everything into memory at once. If your query returns ten million rows, you can fetch them in batches of a thousand and process each batch before requesting the next.

Cursors also serve as a natural boundary for Python's **context manager** protocol. Using the `with` statement — `with connection.cursor() as cur:` — ensures that the cursor is properly closed and its resources released even if an exception occurs mid-query. This pattern, combined with the connection itself being a context manager, gives you a clean, exception-safe way to write database code that does not leak connections or leave dangling transactions. Mastering cursor management is the difference between a script that works once and a script that you can run in production every day without surprises.

## Where SQL + Python Integration Appears in Real Life

The ability to connect Python to a database is not an academic exercise — it is the backbone of almost every data-intensive application in the industry. In **e-commerce**, recommendation engines pull user browsing history and purchase patterns directly from product databases, process them through collaborative filtering models in Python, and write personalised recommendations back into the same database — all within a single scheduled pipeline. In **fintech**, fraud detection systems connect to transaction databases, stream recent activity through anomaly detection models, and flag suspicious transactions in near real time, with the entire loop running inside a Python script that reconnects to the database on every execution.

In **healthcare**, clinical research teams write Python scripts that connect to electronic health record (EHR) databases, extract de-identified patient cohorts using complex SQL queries with multiple inclusion and exclusion criteria, and load the results directly into statistical analysis libraries for survival modelling or drug efficacy studies. The regulatory requirement for auditability means every step of the data extraction must be logged and reproducible — a task made vastly simpler when the entire pipeline lives in a single Python file rather than a chain of manual exports. In **marketing analytics**, automated reporting systems query attribution databases every morning, run cohort analysis and ROAS calculations in Python, and push the results into Tableau or PowerBI dashboards without a human touching a single CSV file. And in **logistics and supply chain**, warehouse inventory databases are queried by Python scripts that forecast demand, optimise restocking schedules, and simulate the impact of supply disruptions — all powered by the same pattern you will learn in this session.

## What's Next

After this session, you will be able to:
- Connect to a SQLite database using Python's built-in `sqlite3` module and execute queries
- Construct and parse connection strings for PostgreSQL authentication
- Execute SQL queries through a cursor and retrieve results into Python data structures
- Use SQLAlchemy Core to write database-agnostic queries that work across engines
- Load SQL query results directly into a Pandas DataFrame for analysis and visualisation
- Manage database transactions with commit and rollback for safe state changes

You do not need to memorise every database driver's API right now — the patterns are almost identical across all of them. The goal is to see your database not as a separate silo, but as a data source your Python code can talk to as naturally as it reads a CSV file.

## Interesting Questions for the Live Session

- If you connect to a database, run a query that returns a million rows, and load it all into a Pandas DataFrame, what happens to your computer's RAM — and how would you design a pipeline that avoids this bottleneck?
- SQLAlchemy lets you write the same Python code for SQLite, PostgreSQL, and MySQL — but if every database has its own SQL dialect and feature set, where does the abstraction break down?
- A cursor can fetch one row, many rows, or all rows at once. In what real-world scenario would fetching one row at a time be preferable to fetching the entire result set?
- If a colleague commits code with a plain-text database password inside the connection string, what are the security implications — and how would you refactor that code to be production-safe?

By the end of this session, the database should feel less like a separate tool you export from and more like a programmable data source your Python code orchestrates on demand: **SQL is not the wall — it is the wire.**
