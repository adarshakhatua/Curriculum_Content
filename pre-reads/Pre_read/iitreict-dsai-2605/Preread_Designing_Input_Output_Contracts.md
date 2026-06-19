# Pre-read: Designing Input-Output Contracts

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1: Programming &amp; Data Foundations (Weeks 1-4)</b><br/><i>(Python, CSV/JSON)</i><br/>Core programming and data handling skills])
    P2([Previous Module<br/><b>Module 2: Data Analysis with NumPy &amp; Pandas (Weeks 5-8)</b><br/><i>(NumPy, Pandas)</i><br/>Numeric computation and tabular analysis])
    P3([Previous Module<br/><b>Module 3: SQL for Data Science (Weeks 9-12)</b><br/><i>(SQL, Database Design)</i><br/>Structured data storage and retrieval])
    P4([Previous Module<br/><b>Module 4: Statistics &amp; Data Visualization (Weeks 13-16)</b><br/><i>(Statistics, BI Tools)</i><br/>Statistical reasoning and visual storytelling])
    P5([Previous Module<br/><b>Module 5: Applied Machine Learning (Weeks 17-20)</b><br/><i>(Scikit-learn, ML Models)</i><br/>Building and evaluating predictive models])
    C1[[Current Module Until Previous Session<br/><b>Module 6: GenAI for Data Science</b><br/><i>(OpenAI API, Prompt Engineering)</i><br/>LLM fundamentals and API integration]]
  end

  CS{{Current Session<br/><b>Designing Input-Output Contracts</b><br/><i>From free text to structured data contracts</i><br/>JSON mode · structured outputs · error handling}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Bridges LLMs to Production</b><br/>Structured contracts let data pipelines safely consume AI outputs, powering the capstone and beyond.])
    RV([Real-Life Value<br/><b>Powers Production AI Apps</b><br/>The difference between a demo script and an enterprise AI integration that downstream systems can trust.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Capstone Project (Weeks 25-26)</b><br/><i>(Streamlit, Full Pipeline)</i><br/>End-to-end AI solution development])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;Data Tools&nbsp;| P3
  P3 ==>|&nbsp;Query&nbsp;| P4
  P4 ==>|&nbsp;Stats&nbsp;| P5
  P5 ==>|&nbsp;ML&nbsp;| C1
  C1 ==>|&nbsp;API&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Capstone&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2,P3,P4,P5,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px;
```

You just shipped a feature that calls an LLM to classify customer support tickets. The first week is magical — the model writes back friendly, accurate labels. Then your product manager asks for a daily CSV of every ticket, the assigned category, and a confidence score. You write a quick script to collect the responses and discover the problem: one day the model replies with `"category: billing"`, the next with `{"category": "billing", "confidence": "high"}`, and the next with `The category for this ticket is billing.` Each is correct English. Each breaks your parser in a different way. Your elegant pipeline is now duct-taped together with regex patches and if-else chains.

This is the hidden tax of working with LLMs — models produce language, not data. Language is flexible, expressive, and maddeningly inconsistent. When you need a program to consume the output, every variation is a bug waiting to happen. You could spend hours writing parsing logic for every edge case, or you could solve the problem at its source: design a contract that tells the model exactly how to reply before it opens its virtual mouth.

That is where **Designing Input-Output Contracts** becomes essential. This session teaches you to define, enforce, and gracefully handle structured communication with LLMs — turning the wild frontier of free-text generation into a disciplined, production-ready API.

What if you could call an LLM the same way you call a REST endpoint — send a request, get back a precisely typed JSON object, and never worry about parsing? What if your data pipeline could process thousands of AI-generated responses without a single `try: except:` block for malformed text? What if you could tell your product manager, with a straight face, that the AI integration has a schema, a validation layer, and a retry policy? This session is the key that turns "the model mostly works" into "the model is an engineerable component of your system."

At its core, an **input-output contract** is a formal agreement between you and the language model. You specify exactly what structure the output must follow — the fields, their types, the allowed values — and the model commits to producing output that fits that mold. The central tool for this is **JSON mode**, a feature of modern LLM APIs that forces responses into valid JSON matching a schema you provide. But the contract is more than just a format switch; it is a design pattern that encompasses **system messages** (where you define the rules), **user messages** (where you supply the data), **structured output schemas** (the blueprint for responses), and **error handling** (what happens when the model still gets it wrong). Think of it like a web form with server-side validation — except the server is a probabilistic text generator, and you have to design the form so creatively that the generator cannot help but fill it out correctly.

In the **previous session**, 22.1, you learned to authenticate with the OpenAI API, send chat completion requests, and tune parameters like temperature and top-p to control randomness. You saw that an API call returns a JSON wrapper with the model's text inside, but you treated the inner content as a black box — you read it, printed it, moved on. Now you will crack that black box open. The authentication dance and parameter tuning you mastered become the scaffolding for a much more ambitious goal: not just getting an answer, but getting an answer that your code can trust, validate, and pipeline into the next stage of processing without human review.

In this pre-read, you will discover:
- How to **configure** JSON mode in an LLM API call to enforce structured responses
- How to **design** system messages that act as a formal contract for output formatting
- How to **apply** structured output schemas to extract data with predictable shapes
- How to **build** error-handling strategies that make LLM calls resilient and auditable

---

## Why JSON Mode Is Non-Negotiable for AI-Working Pipelines

When you ask an LLM to "return the sentiment as positive, negative, or neutral," the model interprets that as a stylistic suggestion, not a constraint. It will happily return any of: `Positive`, `positive`, `"POSITIVE"`, `sentiment: positive`, `{"sentiment": "positive"}`, or even a paragraph explaining why it chose positive. Human readers handle this fine; `split(",")` does not. **JSON mode** solves this by telling the API to emit only valid JSON and to match a schema you supply through the system message. The model's sampling process is constrained at the token level — it cannot output text that would break JSON validity. This shifts the problem from "how do I parse this?" to "did the model populate the fields correctly?" — a far smaller, more tractable question. You still need to validate the values, but you no longer need to guess the format.

Consider an e-commerce pipeline that classifies product returns into reasons (damaged, wrong item, changed mind, etc.). Without JSON mode, you get back: `This item was returned because it was damaged during shipping.` — a sentence. With JSON mode and a schema specifying `{"reason": string, "confidence": float}`, you get `{"reason": "damaged_during_shipping", "confidence": 0.94}` — a row ready to insert directly into a database. The difference is not cosmetic; it is the difference between a system that requires human review and one that runs unattended for weeks.

## System vs. User Messages: Designing a Two-Part Contract

The **system message** is where your contract lives. It is the only place in the API where you can unilaterally define rules that the model cannot override through the conversation. This makes it the natural home for your output schema, formatting instructions, and behavior guardrails. The **user message**, by contrast, carries the dynamic payload — the specific text, question, or data that varies with each call. Separating these two roles is the architectural insight of input-output contracts: the system message says "this is how you must always respond," and the user message says "this is what you must respond about."

A well-designed contract looks like this. Your system message specifies: "You are a classification assistant. Your output must be valid JSON with exactly two fields: `category` (one of: billing, technical, account) and `confidence` (a float between 0 and 1). Do not include any text outside the JSON object." Your user message is simply: "Classify the following ticket: 'I was charged twice for my last payment.'" The model then has no room to negotiate format — it produces exactly `{"category": "billing", "confidence": 0.97}`. If it deviates, your validation layer catches it and triggers a retry. This two-part contract turns every LLM call into a deterministic interface with a known output signature.

## Where Input-Output Contracts Appear in Real Life

Financial services firms use structured contracts to extract data from earnings call transcripts — the system message defines a schema with fields like `revenue`, `net_income`, `eps`, `forward_guidance`, and the user message supplies the transcript text. The output plugs directly into quantitative models without human parsing. In healthcare, clinical notes are fed through LLMs with contracts that produce structured `diagnosis`, `medication`, `follow_up_date` objects, feeding electronic health record systems with minimal friction. E-commerce platforms use the pattern for product categorization at scale: a product title and description go in, a structured object with `category_id`, `attributes`, and `confidence_scores` comes out, ready for database ingestion. Customer support automation pipelines rely on contracts to classify tickets, extract urgency, and route to the correct team — all without a human reading the body. Legal tech companies parse contracts themselves (the irony is intentional) using nested JSON schemas to identify clauses, parties, termination dates, and obligation types. In every case, the pattern is identical: free-text input, structured-contract output, validated pipeline. The industries differ, but the architecture does not.

## What's Next

After this session, you will be able to:
- Configure JSON mode in an LLM API call to enforce structured responses from any supported model
- Separate system instructions from user prompts to build maintainable, testable input-output contracts
- Design structured output schemas that map directly to your application's data models and database tables
- Implement validation and retry logic that catches contract violations and recovers gracefully
- Parse structured LLM responses directly into Pandas DataFrames for downstream analysis
- Build a reusable contract wrapper that abstracts away parsing, validation, and error handling from your business logic

You do not need to memorise every API parameter or error code right now. The goal is to see that an LLM call is not a mystical incantation — it is a function call with a contract, and you are the one who writes the contract.

## Interesting Questions for the Live Session

- When would you choose JSON mode over a plain-text prompt with format instructions, and what are the trade-offs in flexibility versus reliability?
- If an LLM returns structurally valid JSON but semantically wrong values, does your contract hold — and where should you draw the boundary between parsing errors and reasoning errors?
- How would you design a contract for a multi-step reasoning chain where the output of one LLM call becomes the system message or input data for the next?
- What happens to your error-handling strategy when the LLM provider changes its response format — how do you build a contract that survives API version upgrades?

By the end of this session, input-output contracts should feel less like an API detail and more like a design discipline: **You are not prompting — you are programming an unreliable teammate with a structured spec.**
