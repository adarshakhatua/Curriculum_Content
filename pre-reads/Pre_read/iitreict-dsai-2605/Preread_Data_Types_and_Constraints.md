# Pre-read: Data Types & Constraints

## Context of This Session in the Course

%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1(["Previous Module<br/><b>Programming &amp; Data Foundations</b><br/><i>(Python, CSV/JSON/APIs)</i><br/>Built programming foundation with Python"])
    P2(["Previous Module<br/><b>Data Analysis with NumPy &amp; Pandas</b><br/><i>(NumPy, Pandas)</i><br/>Mastered data manipulation with NumPy/Pandas"])
    C1[["Current Module Until Previous Session<br/><b>SQL for Data Science</b><br/><i>(SQL, Database Schemas)</i><br/>Basics of tables, keys and SELECT queries"]]
  end

  CS{{"Current Session<br/><b>Data Types &amp; Constraints</b><br/><i>Data integrity mindset</i><br/>Numeric · String · Date · NULL · NOT NULL · UNIQUE · DEFAULT"}}

  subgraph Value["Why This Matters"]
    CV(["Course Value<br/><b>Foundation for database integrity</b><br/>Ensures clean, trustworthy data for all downstream analysis and ML models."])
    RV(["Real-Life Value<br/><b>Design production-ready databases</b><br/>Build e-commerce catalogs or user systems where correctness is non-negotiable."])
  end

  subgraph Future["Where This Leads Next"]
    F1(["Upcoming Module<br/><b>Statistics &amp; Data Visualization</b><br/><i>(Statistics, Seaborn/Tableau)</i><br/>Statistical analysis and visual storytelling"])
    F2(["Upcoming Module<br/><b>Applied Machine Learning</b><br/><i>(Scikit-learn, ML models)</i><br/>Build predictive machine learning models"])
    F3(["Upcoming Module<br/><b>GenAI for Data Science</b><br/><i>(LLMs, OpenAI API, RAG)</i><br/>Generative AI for data science workflows"])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| C1
  C1 ==>|&nbsp;Constraints&nbsp;| CS
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

You have just been handed a new customer dataset for analysis. The first thing you notice: the age column contains values like -7, 342, and "forty-two". The email column is empty for a third of the rows, and the registration_date column has entries like "last week" and "2024/13/01". Your dashboard refresh is scheduled for tomorrow morning, and the ML pipeline depends on this dataset running clean tonight.

Writing a one-off Python script to scrub the data after it lands in the database feels like the obvious fix — and many teams do exactly that. But each script is a band-aid that leaves the root cause untouched. The next batch arrives with the same problems, your pipeline breaks again, and soon nobody trusts the numbers. You end up fighting symptoms instead of designing a system where bad data never gets in. That is where **Data Types & Constraints** becomes essential.

What if you were asked to design the database schema for a hospital's patient management system? One table stores medication dosages, another stores lab results, and a third tracks appointment history. A single misplaced decimal in the dosage column or a missing patient ID in the appointments table could have real consequences. After this session, you will know exactly how to define columns that reject impossible values, enforce required fields, and prevent duplicates — not through code that might fail, but through the database itself, every single time.

Think of a database column as a labelled container. A **data type** defines what shape the container is — a square peg can only hold square values. An **INTEGER** column rejects text; a **DATE** column rejects anything that is not a valid calendar date; a **VARCHAR(50)** column caps text at 50 characters so nobody stuffs an entire novel into a name field. **Constraints** take this one step further: they are rules that the database checks before it lets any data in. **NOT NULL** says "this field must always have a value." **UNIQUE** says "no two rows can share this value." **DEFAULT** fills in a sensible value when the user provides none. Together, data types and constraints form the immune system of your database — they intercept bad data at the door before it ever reaches your analysis.

In the **previous session**, you learned to retrieve specific data from a database using SELECT, WHERE, DISTINCT, and ORDER BY. You understood that data lives in tables with rows and columns, and that primary and foreign keys connect those tables into a coherent schema. That skill is powerful, but it assumes the data you are querying is trustworthy. A SELECT statement that filters on age > 18 becomes meaningless if the age column contains the string "unknown" or a NULL value. The previous sessions gave you the key to the library; this session teaches you how to ensure the books inside are accurate and well-structured before you ever open one.

In this pre-read, you will discover:
- How to **understand** the differences between numeric, string, and temporal data types and why each choice matters for data integrity.
- How to **apply** NOT NULL and UNIQUE constraints to enforce business rules directly at the schema level.
- How to **recognise** the behaviour of NULL values in calculations and how DEFAULT provides a clean fallback.
- How to **connect** constraint-driven schema design to building reliable analytics and machine learning pipelines.

---

## Why Your Phone Number Should Never Be Stored as an Integer

Data types seem like a mundane detail until one of them silently corrupts your analysis. Consider phone numbers: a well-meaning developer stores them as an INTEGER column because "phone numbers are numbers." The first problem surfaces immediately — leading zeros vanish, turning 08012345678 into 8012345678. The second problem appears at scale: what happens when an international number includes a country code prefix like +91, or when an extension is appended? The INTEGER type cannot store the plus sign or the dash, so the data gets truncated or rejected on insertion. Suddenly, thousands of customer records are missing digits, and your CRM campaign sends SMS messages to wrong numbers.

The same principle applies to every data type decision. Storing currency values as FLOAT instead of DECIMAL leads to rounding errors that compound over thousands of transactions — a problem notorious in financial systems. Storing dates as VARCHAR forces you to parse strings every time you need to sort chronologically or compute the days between two events, and nothing stops a user from typing "next Tuesday" into a field meant for 2024-06-20. The data type is not a suggestion; it is the first contract between your application and your database. Choosing wisely at schema design time saves hours of cleanup later and prevents entire categories of bugs from ever reaching production.

## How Constraints Make the Database Your Most Reliable Colleague

A data type prevents a string from entering an integer column, but it cannot prevent a user from leaving a critical field blank. That is where constraints step in. **NOT NULL** is the simplest and most impactful constraint: it guarantees that a column always holds a value. In a user table, the email column should almost certainly be NOT NULL — without it, your authentication system breaks and your marketing campaigns fail. **UNIQUE** goes further by guaranteeing that every value in a column is distinct. A UNIQUE constraint on an email column means the database itself rejects duplicate sign-ups, no matter how many times a user clicks the submit button. You do not need to write a single line of application code to check for duplicates.

**DEFAULT** handles the opposite problem: what value should the database use when the user provides nothing? Instead of letting a column sit NULL — which can silently break arithmetic and comparisons — you can set a DEFAULT. For example, a "subscription_status" column with DEFAULT 'active' ensures that every new user starts with a valid status. A "created_at" column with DEFAULT CURRENT_TIMESTAMP stamps every row automatically. These constraints are not theoretical niceties; they are the difference between a database that needs constant babysitting and one that runs itself. Together, NOT NULL, UNIQUE, and DEFAULT form a three-line defence that catches the vast majority of data quality issues before they ever touch your analysis.

## Where Data Types & Constraints Appear in Real Life

In **e-commerce**, product catalogs rely on UNIQUE constraints on SKU codes to prevent two products from sharing the same identifier, and DECIMAL types for prices to avoid the rounding errors that would accumulate across millions of transactions. A NOT NULL constraint on the inventory count ensures that a product is never listed with an unknown stock level — and a DEFAULT of 0 prevents null pointers in the shopping cart logic. In **healthcare**, patient records demand DATE types for dates of birth (so nobody is born on February 30th) and VARCHAR types with CHECK constraints on blood type to allow only valid values like A+, B-, or O+. A NOT NULL constraint on the patient ID in an appointments table guarantees that every booking belongs to someone real.

In **financial services**, account balances use DECIMAL with exact precision — never FLOAT — because a rounding error of a few cents multiplied across thousands of transactions becomes a compliance audit nightmare. Transaction tables use UNIQUE on transaction reference numbers and NOT NULL on every monetary column, because a missing amount in a wire transfer is not a data quality issue; it is a legal liability. In **SaaS platforms**, user tables enforce UNIQUE on email addresses, DEFAULT 'active' on account status, and DEFAULT CURRENT_TIMESTAMP on created_at — ensuring every new sign-up is immediately queryable and traceable. In **logistics and supply chain**, tracking numbers carry UNIQUE constraints, origin and destination columns are NOT NULL, and shipment dates use the DATE type so that sorting by delivery date works correctly out of the box. In every case, the pattern is the same: define the rules at the schema level once, and the database enforces them on every single row, forever.

## What's Next

After this session, you will be able to:
- Choose the appropriate SQL data type for any column based on the nature of its values — numeric, textual, or temporal.
- Apply NOT NULL constraints to guarantee that critical columns always contain a value.
- Enforce uniqueness across rows with the UNIQUE constraint to prevent duplicate entries.
- Set DEFAULT values to handle missing input gracefully without relying on application code.
- Design a table schema that balances data integrity with real-world flexibility.
- Identify and explain data type and constraint violations when debugging a broken pipeline.

You do not need to memorise every SQL data type variant right now. The goal is to think of your database schema as a contract: once you define the rules upfront, the database enforces them automatically, every single time.

## Interesting Questions for the Live Session

- If a phone number column is defined as INTEGER, what data is silently lost compared to VARCHAR — and could that ever cause a production bug that goes undetected for months?
- Why might a database designer intentionally allow NULLs in a column instead of providing a DEFAULT — when is that flexibility worth the added risk to data quality?
- If a UNIQUE constraint prevents duplicate rows, how would you design a table where the same customer can place multiple orders but never two orders with the same order number?
- Consider a "date_of_birth" column defined as DATE. What happens when your sign-up form sends '1900-01-01' as a placeholder for unknown birthdays — is that a data type failure, a constraint failure, or something else entirely?

By the end of this session, data integrity should feel less like an abstract database concept and more like a practical design tool: **the database becomes your most reliable code reviewer.**
