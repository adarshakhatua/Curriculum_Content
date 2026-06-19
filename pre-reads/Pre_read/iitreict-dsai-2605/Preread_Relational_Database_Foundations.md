# Pre-read: Relational Database Foundations

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Programming &amp; Data Foundations</b><br/><i>(Python, REST APIs)</i><br/>Built core Python and data literacy])
    P2([Previous Module<br/><b>Data Analysis with NumPy &amp; Pandas</b><br/><i>(NumPy, Pandas)</i><br/>NumPy, Pandas, and data exploration])
  end

  CS({{Current Session<br/><b>Relational Database Foundations</b><br/><i>Think in tables, not scripts</i><br/>Tables · Rows · Columns · Keys · Schema Design}})

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Bridges data tools to real-world databases</b><br/>Mastering relational databases gives you the ability to query and organize data at scale — a skill every future module depends on for sourcing, joining, and preparing real datasets.])
    RV([Real-Life Value<br/><b>The backbone of data-driven companies</b><br/>Every company stores its customer, product, and transaction data in relational databases — knowing how they work is what separates someone who can analyze data from someone who can access it in the first place.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Statistics &amp; Data Visualization</b><br/><i>(Stats, Seaborn)</i><br/>Statistical reasoning and visual communication])
    F2([Upcoming Module<br/><b>Applied Machine Learning</b><br/><i>(Scikit-learn, ML)</i><br/>Predictive modeling with Scikit-learn])
    F3([Upcoming Module<br/><b>GenAI for Data Science</b><br/><i>(LLMs, RAG)</i><br/>LLMs, embeddings, and AI applications])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1
  F1 ==>|&nbsp;Analyze&nbsp;| F2
  F2 ==>|&nbsp;Predict&nbsp;| F3

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2 previous;
  class CS current;
  class CV,RV value;
  class F1,F2,F3 future;
  linkStyle default stroke-width:3px
```

You have just joined the analytics team at a fast-growing e-commerce company. On your first day, your manager hands you a CSV file with 2 million rows of customer transactions — but when you ask about the data dictionary, she points to a whiteboard with boxes and arrows connecting tables called `Customers`, `Orders`, `Products`, and `Payments`. The entire company operates on this web of interconnected tables, yet the only way you have ever worked with data is flat files in Python or Pandas.

Opening the CSV, you quickly realise the data is denormalised — product names repeat thousands of times, customer addresses are scattered across rows, and there is no way to tell which transactions belong to the same customer without manually cross-referencing columns. The naive approach of loading everything into a single DataFrame leads to memory bloat, inconsistent updates, and fragile analysis that breaks the moment someone adds a new product. You need a way to structure data that eliminates redundancy, guarantees consistency, and lets you ask questions across multiple tables without copying data around. That is where **Relational Database Foundations** becomes essential.

What if you could walk into any company — a bank, a hospital, a social media platform — and immediately understand how its core data is organised, simply by looking at a handful of table definitions? What if you could answer questions like "Which customers have placed more than five orders this quarter?" without writing Python loops or merging DataFrames by hand? This session gives you the mental model of relational design — the same model that powers every major database system in use today — so that structured data stops being a mystery and becomes a tool you can reason about with clarity.

A relational database is fundamentally a collection of **tables**, where each table represents a single concept — customers, orders, products — and every **row** in that table is a single instance of that concept. Each **column** captures one attribute: a customer's name, an order date, a product price. What makes this design powerful is not the tables themselves but the relationships between them. A **primary key** is a column (or combination of columns) that uniquely identifies every row in a table — no two customers share the same customer ID. A **foreign key** is a column in one table that points to the primary key of another, creating a link: an order row carries a `customer_id` that refers back to the `Customers` table.

Think of it like a well-organised library. Books are shelved by category (the tables), each book has a unique call number (the primary key), and a checkout record references that call number rather than copying the author and title every time someone borrows the book (the foreign key). If the library copied full book details onto every checkout slip, the system would be impossible to maintain — change the author's name once and you would need to update thousands of slips. That is exactly the problem relational **schema design** solves. By separating what something is (a product) from where it appears (an order line), you eliminate duplication and create a single source of truth.

In this session, you will explore **Tables**, **Rows**, **Columns**, **Primary Keys**, **Foreign Keys**, and **Schema Design** — the foundational vocabulary of every SQL database you will use throughout the rest of the course and in your career. These are not abstract concepts; they are the building blocks that turn raw data into a structured, queryable, scalable system.

In the **previous session**, Visualizing Data with Seaborn, you learned to create statistical plots — scatter plots, histograms, heatmaps — to uncover hidden patterns. You became skilled at manipulating data that was already loaded into memory as DataFrames and arrays. But every dataset you worked with arrived as a pre-packaged CSV or API response — someone else had already designed the structure and handed it to you in a single file. This session flips that dependency. Instead of consuming pre-structured flat files, you will learn how data is organised at its source. The NumPy arrays and Pandas DataFrames you grew comfortable with are, in many ways, inspired by the relational model — a table is a DataFrame, a row is a record, a column is a Series. By understanding the database foundations, you will see why Pandas behaves the way it does and, more importantly, how to bridge the gap between the database that stores the data and the Python tooling that analyses it.

In this pre-read, you will discover:
- How to **recognise** the difference between a well-structured table and a flat file that should be split into multiple tables.
- How to **connect** tables using primary and foreign keys to model real-world relationships.
- How to **apply** the principles of schema design to eliminate redundancy and ensure data integrity.
- How to **build** the mental model that turns a blank whiteboard into a production-ready database schema.

---

## Why a Table Is More Than a Spreadsheet Tab

At first glance, a database table looks just like a spreadsheet: rows and columns filled with values. But a table carries a contract that a spreadsheet does not. Every column has a defined **data type** — that column will never accidentally contain a date where a price belongs. Every row is uniquely identifiable, so you can point to exactly one record and say "this one". And unlike a spreadsheet, a table is designed to be one piece of a larger puzzle. A `Customers` table does not repeat order details because it trusts the `Orders` table to hold that information and link back through a shared identifier.

This rigor is what makes databases reliable at scale. When an e-commerce platform processes 10,000 orders per minute, it cannot afford to check whether every spreadsheet row has consistent formatting. The table structure enforces consistency at the storage level. As a data scientist, understanding this means you can trust the data you query — and when something looks wrong, you know whether to question the query or the schema.

## How Primary and Foreign Keys Turn Data Into a Web

Imagine trying to run a hospital without patient IDs. Every time a doctor refers to a patient, they would need to spell out the full name, date of birth, and address — and hope nothing was mistyped. That is the world without **primary keys**. A primary key is the database's way of saying "I know exactly which record you mean." It is typically a single column like `patient_id` or `order_id`, and it is guaranteed to be unique and non-null for every row.

A **foreign key** is the other half of the relationship. It is a column in one table that stores the primary key value of another table. When an `Appointments` table has a `patient_id` column, that column is a foreign key referencing the `Patients` table. This one mechanism is what allows you to join two tables and ask: "Show me all appointments for patients over 65." Without foreign keys, every query would require manual matching and error-prone assumptions about which records belong together. The combination of primary and foreign keys is what transforms isolated tables into a connected, queryable web of data.

## Where Schema Design Appears in Real Life

Schema design is not an academic exercise — it is the daily work of every organisation that manages structured data at scale. In **e-commerce**, every product listing, shopping cart, and order history is governed by a schema that separates product catalogs from inventory from customer accounts. When you search for a product and see "Only 3 left in stock," that number comes from a foreign key lookup between the products table and the inventory table. In **healthcare**, patient records, lab results, and prescriptions are spread across multiple tables linked by patient IDs and encounter IDs — a poorly designed schema here could mean a doctor misses a critical allergy because the data was stored in the wrong place. In **banking**, transaction tables, account tables, and customer tables must be designed so that money never disappears: a withdrawal from one account and a deposit into another are two rows in a ledger table, linked by foreign keys to both accounts. In **social media**, the entire friend graph, news feed, and notification system is a network of relational tables — users, posts, likes, comments — where every like carries a foreign key to both the user and the post. And in **logistics**, package tracking depends on schemas that connect shipments, warehouses, routes, and delivery confirmations across multiple geographic locations. In every case, the quality of the schema determines whether the data is trustworthy, queryable, and scalable — or whether it collapses under its own complexity.

## What's Next

After this session, you will be able to:
- Distinguish between a flat file and a properly normalised relational table structure by inspecting column dependencies.
- Identify primary keys and foreign keys in any database schema diagram and explain the relationship they encode.
- Design a simple multi-table schema for a business domain such as e-commerce or healthcare.
- Trace how data integrity is preserved through primary key uniqueness and foreign key referential constraints.
- Map the relational concepts of tables, rows, and columns to Pandas DataFrames, rows, and Series.

You do not need to memorise SQL syntax right now — that comes next session. The goal is to see the world of data the way a database sees it: as a web of connected tables where every value has a home and every relationship has a meaning.

## Interesting Questions for the Live Session

- If a table does not have a primary key, what specific problems will you encounter when trying to update or delete a single row?
- Can a foreign key column contain a NULL value, and if so, what does that imply about the relationship between the two tables?
- How would you model a situation where one order can contain multiple products, and one product can appear in multiple orders?
- If you were designing a schema for a social media platform, would you store the number of likes on a post as a column in the posts table, or would you compute it from a separate likes table every time? What are the tradeoffs?

By the end of this session, relational databases should feel less like a black box and more like a blueprint: **a table is a concept, a key is a connection, and a schema is a map of how the real world fits together.**
