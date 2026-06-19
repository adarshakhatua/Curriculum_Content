# Pre-read: Basic SQL Querying

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1: Programming &amp; Data Foundations</b><br/><i>(Python, CSV/JSON)</i><br/>Core Python, data formats, and APIs])
    P2([Previous Module<br/><b>Module 2: Data Analysis with NumPy &amp; Pandas</b><br/><i>(NumPy, Pandas)</i><br/>NumPy arrays, DataFrames, and EDA])
    C1[[Current Module Until Previous Session<br/><b>Module 3: SQL for Data Science</b><br/><i>(Relational DBs, Keys/Schema)</i><br/>Tables, keys, and schema design fundamentals]]
  end

  CS{{Current Session<br/><b>Basic SQL Querying</b><br/><i>From storage to retrieval</i><br/>SELECT &middot; WHERE &middot; DISTINCT &middot; LIMIT &middot; ORDER BY &middot; logical operators}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Foundation for every SQL skill</b><br/>Mastering SQL querying is the prerequisite for joins, aggregations, window functions, and Python-SQL integration.])
    RV([Real-Life Value<br/><b>The universal language of data</b><br/>Every data professional uses SQL daily to extract and explore data from production databases.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 4: Statistics &amp; Data Visualization</b><br/><i>(Descriptive Stats, BI Tools)</i><br/>Statistical analysis, hypothesis testing, and dashboards])
    F2([Upcoming Module<br/><b>Module 5: Applied Machine Learning</b><br/><i>(Scikit-learn, Regression)</i><br/>Linear models, ensemble methods, and ML pipelines])
    F3([Upcoming Module<br/><b>Module 6: GenAI for Data Science</b><br/><i>(LLMs, OpenAI API)</i><br/>Prompt engineering, RAG, and AI-powered analytics])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| C1
  C1 ==>|&nbsp;Query&nbsp;| CS
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

You open a database containing millions of customer transactions. You need to find every customer who made more than three purchases last month, lives in a specific city, and has not been contacted in the last 90 days. The database has dozens of tables with hundreds of columns each — and you have exactly one hour to deliver the list.

If you downloaded every row and filtered in a spreadsheet, you would run out of memory before processing even a fraction of the data. Dragging entire tables into Python or R also wastes time and bandwidth — the data you need might be a tiny fraction of what is stored. Without a way to ask the database to do the filtering for you, you are stuck moving massive files around, waiting for your machine to catch up.

That is where Basic SQL Querying becomes essential.

What if you could ask a database a question in plain English and get exactly the rows you need within seconds — no matter how large the dataset? What if you could combine conditions, sort results, and remove duplicates with just a few lines of code? This session gives you that power. By learning the SELECT statement, the WHERE clause, and the other core building blocks of SQL, you will be able to retrieve data from any relational database with confidence and precision.

At its heart, **SQL** — Structured Query Language — is the language you use to talk to relational databases. A relational database stores data in **tables**, which look a lot like spreadsheets: rows represent individual records, and columns represent fields of information. But unlike a spreadsheet, a database is optimised for answering questions — **queries** — without loading everything into memory. The most fundamental command, **SELECT**, tells the database which columns you want to see. The **FROM** clause specifies which table to look in. Together, `SELECT ... FROM ...` is the starting point for every query you will ever write.

Think of a library. The database is the entire library building. Each table is a bookshelf — one shelf for books, another for authors, another for borrowers. Writing a query is like walking up to a librarian and saying, "Please show me the titles and authors of every book published after 2020, sorted by author name, and only the first 10 results." The librarian does not dump every book on your desk — they walk to the relevant shelves, pick the matching books, and hand you exactly what you asked for. That is what SQL does for your data.

In this session, you will go beyond basic retrieval. You will learn to filter rows with the **WHERE clause** using comparison and logical operators (`AND`, `OR`, `NOT`). You will eliminate duplicate rows with **DISTINCT**, control how many rows you see with **LIMIT**, and sort your results with **ORDER BY**. These six tools — SELECT, WHERE, DISTINCT, LIMIT, ORDER BY, and logical operators — form a complete toolkit for answering most data questions you will encounter in the real world.

In the **previous session**, you learned how relational databases are built — how tables are designed with primary keys and foreign keys, how rows and columns form the structure, and how schemas ensure data integrity. That session gave you the blueprint of a database. This session teaches you how to read that blueprint and extract the information stored within it. Understanding how tables relate to each other (via keys) is what makes your SELECT and WHERE clauses meaningful — you cannot effectively retrieve data unless you know where it lives and how it is organised. The schema design knowledge from session 9.1 is the map; Basic SQL Querying is the compass.

In this pre-read, you will discover:
- How to **retrieve** specific columns and rows from a database table using SELECT and FROM
- How to **apply** filtering conditions with WHERE and logical operators to narrow down results
- How to **recognise** and eliminate duplicate data using DISTINCT
- How to **interpret** sorted and limited results with ORDER BY and LIMIT

---

## Why WHERE Is the Most Powerful Word in SQL

A database with a million rows is useless if you cannot narrow it down to the ten rows that matter. The WHERE clause is what turns a data dump into a targeted answer. Without it, every query returns the entire table — a behaviour that might be acceptable for a ten-row configuration table but catastrophic for a transaction log with hundreds of millions of entries.

A WHERE clause filters rows based on a condition. You can use **comparison operators** (`=`, `>`, `<`, `>=`, `<=`, `!=`) to match numeric or date values, and you can chain conditions with **logical operators** like `AND`, `OR`, and `NOT`. For example, `WHERE city = 'Mumbai' AND amount > 1000` returns only rows where both conditions are true. The database evaluates these filters at the storage engine level — it only fetches the matching rows, never loading the entire table into memory. This is the fundamental reason databases can handle terabytes of data while spreadsheets choke on a few hundred thousand rows.

This is the same logic you have already used in Python conditionals and Pandas boolean indexing. The mental model transfers directly: think of WHERE as a selection mask that keeps only the rows where your condition evaluates to true. Mastering this unlocks every future SQL skill — joins, subqueries, aggregations, and window functions all rely on precise filtering to be useful. Every piece of SQL you learn after this session will depend on your ability to write a clean WHERE clause.

## Sorting, Limiting, and Deduplicating — The Polish on Every Query

Once you have the right rows, how do you present them in a useful order? And what if the same value appears a thousand times but you only want to see it once? These are the questions that separate a raw query from a decision-ready answer.

**ORDER BY** sorts your results by one or more columns, in ascending (`ASC`, the default) or descending (`DESC`) order. You can sort by multiple columns — for example, `ORDER BY region ASC, revenue DESC` first groups all rows alphabetically by region, then within each region sorts by revenue from highest to lowest. **LIMIT** chops the result set to a specified number of rows, which is essential when you want a quick preview of a large table or are building paginated reports for a dashboard. **DISTINCT** removes duplicate rows from your result, giving you a clean list of unique values in a column or combination of columns. These three tools turn raw filtered data into a polished table you can act on immediately.

Every dashboard you have ever used relies on these operations behind the scenes. The "Top 10 Products" widget on an e-commerce site? That is `ORDER BY revenue DESC LIMIT 10`. The dropdown of unique categories in a filter panel? That is `SELECT DISTINCT category`. A paginated employee directory showing 25 names per page? That is LIMIT combined with an offset. These are not advanced features — they are the everyday polish that makes data usable, and they are among the first tools you will use in every professional SQL environment.

## Where SQL Querying Appears in Real Life

SQL is not just an academic tool — it is consistently ranked as one of the top two most requested technical skills in data job descriptions, alongside Python. The SELECT-WHERE-ORDER BY-LIMIT pattern you will learn in this session powers data retrieval across nearly every industry.

In **e-commerce and retail**, every product search, customer segmentation report, and inventory analysis starts with a SELECT query. Analysts routinely ask "Show me the top 10 products by revenue in the last quarter" or "List all customers who have not purchased in 90 days". These are the exact queries you will learn to write. In **healthcare and life sciences**, medical researchers query patient databases to find cohorts for clinical trials — filtering for patients with specific conditions, age ranges, and prior treatments. Insurers use SQL to detect fraudulent claims by filtering for unusual billing patterns. In **finance and banking**, risk analysts retrieve transactions exceeding a threshold, sorted by amount, to flag suspicious activity for compliance review. Portfolio managers query holdings data with DISTINCT to identify unique asset classes across thousands of accounts. In **marketing and advertising**, teams pull campaign performance data daily — impressions by channel, click-through rates by region, top-performing ad creatives — every one of these queries built on the SELECT-WHERE-ORDER BY-LIMIT foundation. Each industry speaks its own domain language, but the SQL dialect remains the same: state what you need, filter what you do not, and let the database deliver.

## What's Next

After this session, you will be able to:
- Write a SELECT query to retrieve specific columns from any database table
- Filter rows using the WHERE clause with comparison and logical operators
- Remove duplicate results using DISTINCT
- Sort query results with ORDER BY in ascending and descending order
- Limit the number of returned rows using LIMIT
- Combine all of these clauses into a single, precise query

You do not need to memorise every SQL function right now. The goal is to make a habit of thinking in queries before you write code: **State what you need, filter what you do not, and let the database do the heavy lifting.**

## Interesting Questions for the Live Session

- What happens when you combine ORDER BY with a column that contains NULL values — do NULLs sort first or last, and why does that matter for real-world reporting?
- If you use SELECT DISTINCT on a table with a million rows, does the database scan every row or can it use an index to speed things up — and how would you find out?
- When would you intentionally write a WHERE clause that returns zero rows, and can that ever be a useful debugging technique?
- How would the performance of `WHERE amount > 1000 OR amount < 500` differ from `WHERE amount > 1000 OR status = 'active'` — and what does that reveal about how databases evaluate OR conditions?

By the end of this session, SQL should feel less like a syntax exercise and more like a direct conversation with your data: **Ask precisely, get exactly what you need.**
