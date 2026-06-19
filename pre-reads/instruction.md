# Pre-read Generation — Master Instruction

This file is the single source of truth for generating student pre-reads for any session of any course in this repository.
Follow every section in order. Do not skip steps. Do not paraphrase instructions — execute them exactly.

---

## INPUTS REQUIRED

Before starting, you must have the following three inputs. If any are missing, ask for them before proceeding.

```
COURSE_NAME   : The exact course name or cohort identifier provided by the user.
                Examples: "iitreict-dsai-2603", "iitreict-dsai-2605"

SESSION_TITLE : The exact title of the session for which the pre-read is being created.
                Example: "Numeric Manipulation & Linear Algebra Basics"

SESSION_LOs   : The Learning Objectives for this session (1–4 bullet points describing what
                students will be able to do after the session).
                These may be taken from the CSV or provided directly by the user.

SESSION_TOPICS: The key topics covered in this session (comma-separated list or short bullets).
                These may be taken from the CSV or provided directly by the user.
```

---

## STEP 1 — LOCATE THE CORRECT COURSE SYLLABUS

### 1.1 — Find the Right CSV File

All course syllabi are stored as CSV files inside:
```
Course_Syllabus/
```

Each CSV filename contains the course/cohort identifier. Match `COURSE_NAME` against the filenames in that folder using a **partial, case-insensitive match**.

**Matching logic:**
- Normalise both the `COURSE_NAME` and each filename: lowercase, strip leading/trailing whitespace.
- Match on cohort ID substring (e.g. `"2603"` matches any filename containing `"2603"`).
- If the user provides an institution prefix (e.g. `"iitp"`, `"iitreict"`), use it as an additional filter to disambiguate.
- If no match is found, list all available CSV filenames and ask the user to confirm which one to use.
- If multiple files match, display the matches, pick the most specific, and confirm with the user before proceeding.

### 1.2 — Detect and Normalise the CSV Schema

Different courses use different column names and grouping structures. Before parsing, detect which schema the file uses, then normalise it into the **Standard Internal Schema** defined below.

#### Known Schemas

| Schema ID | Grouping Column | Title Column | LO Column | Topics Column | Used By |
|---|---|---|---|---|---|
| **A** | `Module` (named string, spans rows — empty cells inherit) | `Title` | `Objective` | `Topics Covered` | iitreict-dsai-2603, iitreict-dsai-2605 |
| **B** | `Module` (named string, present on every row) | `Session Title` | `Learning Objective` | `Topics Covered (2-hr Content Scope)` | iitp-aiml-2601 |
| **C** | `Trimester` (named string, present on every row) | `Session Title` | `Objective` | `Topics Covered` | iitp-aiml-2510 |
| **D** | `Module` (numeric integer: `1`, `2`, `3` — present on every row, no names) | `Title` | `Learning Objective/s of the session (S13n)` | `Topics Covered` | iitp-aiml-2604 |

#### Detection Rules

1. Read the header row of the CSV.
2. If the first column header is `Trimester` → **Schema C**.
3. Else if the header contains `Session Title` and `Learning Objective` → **Schema B**.
4. Else if the header contains `Learning Objective/s of the session (S13n)` → **Schema D**.
5. Else if the first column is `Module` and `Title` is used → **Schema A**.
6. If none match, print the header row and ask the user which fields map to Title, LO, and Topics.

#### Standard Internal Schema (normalise all files to this)

After detection, map every CSV into these unified field names for all downstream steps:

```
GROUP_NAME  : The grouping unit (Module name or Trimester name)
              — For Schema A: carry forward non-empty Module cell values downward
              — For Schema B: Module column (already on every row)
              — For Schema C: Trimester column (already on every row)
              — For Schema D: construct a name as "Module {N}" from the numeric value

WEEK        : Week number (present in all schemas as `Week`)
SESSION_ID  : Session identifier (present in all schemas as `Session`)
TITLE       : Session title
              — Schema A, D: `Title` column
              — Schema B, C: `Session Title` column
LO          : Learning Objective
              — Schema A, C: `Objective` column
              — Schema B: `Learning Objective` column
              — Schema D: `Learning Objective/s of the session (S13n)` column
TOPICS      : Topics covered
              — Schema A, C, D: `Topics Covered` column
              — Schema B: `Topics Covered (2-hr Content Scope)` column
```

> **Schema A carry-forward rule**: Rows where `Module` is empty inherit the `GROUP_NAME` from the most recent non-empty `Module` cell above them. Apply this fill-down before any further processing.

> **Schema D module naming**: The `Module` column contains bare integers (`1`, `2`, `3`). Construct a display label as `"Module {N}"` (e.g. `"Module 1"`, `"Module 2"`). Do not invent module names — if the CSV does not provide a descriptive name, use this numeric label in the Mermaid diagram.

> **LO field in Schema D**: The LO column may be empty for many rows (populated only for specific key sessions). If the LO field is empty for the target session, use the `Topics Covered` column content as a proxy for the session objective.

Load all rows after normalisation. Each row is one session. Discard any row where `TITLE` is empty or `TITLE` equals the GROUP_NAME (these are grouping header artefacts, not real sessions).

---

## STEP 2 — BUILD THE COURSE MAP

From the parsed CSV, construct a **structured course map** in your working memory:

```
[
  {
    module_name: "Module 1: Programming & Data Foundations (Weeks 1-4)",
    sessions: [
      { session_id: "1.1", title: "...", objective: "...", topics: "..." },
      { session_id: "1.2", title: "...", objective: "...", topics: "..." },
      ...
    ]
  },
  {
    module_name: "Module 2: ...",
    sessions: [ ... ]
  },
  ...
]
```

### 2.1 — Locate the Target Session

Find the row in the course map where `title` exactly matches `SESSION_TITLE` (case-insensitive).

From this row, extract:
- `target_session_id` — e.g. `"5.3"`
- `target_module_name` — e.g. `"Module 2: Data Analysis with NumPy & Pandas (Weeks 5–8)"`
- `target_module_index` — which module number this is (1, 2, 3…)
- `target_session_objective` — use `SESSION_LOs` if provided by user, else use the CSV `Objective` column
- `target_session_topics` — use `SESSION_TOPICS` if provided by user, else use the CSV `Topics Covered` column

### 2.2 — Identify the Surrounding Context

Using the course map, identify:

**A. Previous modules (up to 3):**
- All modules that appear before `target_module_name` in the course map.
- Take the **3 most recent** modules before the current module.
- If there are fewer than 3, take all of them.
- For each, extract: module name + the 2 most representative tech stacks or concepts + a 4–6 word summary of what was learned.

**B. Current module — sessions before the target:**
- All sessions in `target_module_name` that appear before `target_session_id`.
- Summarise what was covered until the previous session in 1 concise phrase (10–15 words max).
- Extract the 2 most representative tech stacks or concepts from those sessions.

**C. Upcoming modules (up to 3):**
- All modules that appear after `target_module_name` in the course map.
- Take the **next 3** upcoming modules only.
- If fewer than 3 exist, take all of them.
- For each, extract: module name + the 2 most representative tech stacks or concepts + a 4–6 word summary of what lies ahead.

**D. Session value:**
- **Course value**: How does mastering this session unlock or strengthen future modules in the course? Write 1 crisp sentence (max 10 words as the heading, 1 sentence as the body).
- **Real-life value**: What industry use case or professional skill does this session enable? Write 1 crisp sentence (same format).

---

## STEP 3 — BUILD THE MERMAID MENTAL MODEL DIAGRAM

Use the context from Step 2 to construct a Mermaid `flowchart TB` diagram. Follow every rule below precisely.

### 3.1 — Diagram Structure Rules

```
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    [Previous module boxes] [Current module until previous session box]
  end

  [Current session box — highlighted]

  subgraph Value["Why This Matters"]
    [Course value box] [Real-life value box]
  end

  subgraph Future["Where This Leads Next"]
    [Upcoming module boxes]
  end

  [Arrow connections]
  [classDef declarations]
  [class assignments]
  linkStyle default stroke-width:3px
```

### 3.2 — Box Construction Rules

**Previous module boxes** (`P1`, `P2`, `P3`):
- Shape: `([ ])` — rounded stadium
- Label format:
  ```
  Previous Module<br/><b>Module Name</b><br/><i>(TechStack1, TechStack2)</i><br/>Short summary phrase
  ```
- Use at most 3. Use variable names `P1`, `P2`, `P3`.

**Current Module Until Previous Session box** (`C1`):
- Shape: `[[ ]]` — subroutine box (double-border rectangle)
- Label format:
  ```
  Current Module Until Previous Session<br/><b>Module Name</b><br/><i>(TechStack1, TechStack2)</i><br/>Short summary phrase
  ```
- Use variable name `C1`.

**Current Session box** (`CS`):
- Shape: `{{ }}` — hexagon (highlighted, makes it stand out)
- Label format:
  ```
  Current Session<br/><b>Session Title</b><br/><i>Mental shift or key lens</i><br/>Key concepts · separated · by · middle dots
  ```
- Use variable name `CS`.
- This is the most important box — make the label clear and energising.

**Course Value box** (`CV`) and **Real-Life Value box** (`RV`):
- Shape: `([ ])` — rounded stadium
- Label format:
  ```
  Course Value<br/><b>Short headline (4–6 words)</b><br/>One-sentence explanation
  ```
  ```
  Real-Life Value<br/><b>Short headline (4–6 words)</b><br/>One-sentence explanation
  ```

**Upcoming module boxes** (`F1`, `F2`, `F3`):
- Shape: `([ ])` — rounded stadium
- Label format:
  ```
  Upcoming Module<br/><b>Module Name</b><br/><i>(TechStack1, TechStack2)</i><br/>Short phrase of what lies ahead
  ```
- Use at most 3. Use variable names `F1`, `F2`, `F3`.

### 3.3 — Arrow Connection Rules

Connect boxes in the following order:

```
P1 ==>|&nbsp;Foundation&nbsp;| P2
P2 ==>|&nbsp;[label]&nbsp;| P3
P3 ==>|&nbsp;[label]&nbsp;| C1
C1 ==>|&nbsp;[label]&nbsp;| CS
CS ==>|&nbsp;Course Path&nbsp;| CV
CS ==>|&nbsp;Real-Life Use&nbsp;| RV
CV ==>|&nbsp;Build&nbsp;| F1
F1 ==>|&nbsp;Scale&nbsp;| F2
F2 ==>|&nbsp;Apply&nbsp;| F3
```

- If there is only 1 previous module: `P1 ==>|&nbsp;Foundation&nbsp;| C1`
- If there are 2 previous modules: chain `P1 → P2 → C1`
- Arrow labels must be 1–3 words, meaningful, not placeholders.
- Always use `&nbsp;` around label text for breathing room.
- Do NOT write `==>|<you add label here>|`.

### 3.4 — Colour and Style Rules

```
classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

class P1,P2,P3,C1 previous;
class CS current;
class CV,RV value;
class F1,F2,F3 future;
```

Only assign `class` statements for nodes that actually exist in the diagram.

### 3.5 — Mermaid Safety Rules

- Always open with:
  ```
  %%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
  ```
- Use `<br/>`, `<b>`, `<i>`, `&amp;`, `&nbsp;` for safe formatting inside labels. Never use raw `&`.
- Do NOT use inline `style=`, `padding:`, `font-size:`, `margin:` inside node labels. These render as visible text.
- Do NOT use `/[ ]/` slant-box shape. It produces artifacts.
- Do NOT include session numbers like "Session 5.3" in the diagram — use positional labels only.
- Output only the Mermaid code block. No explanation before or after it inside this section.

---

## STEP 4 — WRITE THE PRE-READ DOCUMENT

Now compose the full pre-read markdown file. Follow the section structure below exactly — same headings, same order. Every section is mandatory.

Use the following writing standards throughout:
- Write in **second person** ("you", "your") — address the student directly.
- Use **plain, professional English** — no jargon without explanation; no padding sentences.
- Paragraphs should be **3–6 sentences** each. No one-sentence paragraphs.
- **Bold** key terms the first time they appear.
- The tone should be: curious, warm, slightly challenging — like a mentor who believes in the student.
- Do NOT use bullet points in the narrative sections. Only use bullets in the "What's Next" and "In this pre-read you'll discover" sections.

---

### SECTION 1 — Title

```markdown
# Pre-read: {SESSION_TITLE}
```

---

### SECTION 2 — Context Diagram

```markdown
## Context of This Session in the Course
```

Paste the complete Mermaid code block from Step 3 here, wrapped in triple backticks with `mermaid` language tag.

---

### SECTION 3 — Opening Narrative Hook (no heading)

Write **2–3 paragraphs** that:

1. **Paragraph 1 — Real-world scenario**: Open with a vivid, relatable, concrete scenario that a working professional or student would recognise. Make it specific. Do not start with "In today's world" or "In this age of AI". Start with an action or a situation.
2. **Paragraph 2 — The real problem**: Deepen the challenge. Show why the naive or intuitive approach to the problem fails. Introduce the tension that this session resolves.
3. **Paragraph 3 — The bridge sentence**: Introduce the session topic naturally as the answer to the tension. End with a clear "That is where **[Session Topic]** becomes essential."

---

### SECTION 4 — "What If" Challenge (no heading)

Write **1–2 paragraphs** using a "What if..." framing. This section asks the student to imagine a professional challenge they will be capable of handling after this session. The purpose is to create motivation and curiosity before the technical content.

- Start with **"What if"** — literally use that phrase.
- Make the scenario ambitious but realistic.
- End with a sentence that frames the session as the key to unlocking this capability.

---

### SECTION 5 — Conceptual Foundation (no heading)

Write **2–3 paragraphs** that:

1. Explain the session's core concept(s) in plain language. Define each key term clearly the first time it appears in **bold**.
2. Use an analogy or metaphor to make the concept stick.
3. State what specific tools, techniques, or mental models will be explored — referencing `SESSION_TOPICS` naturally within the prose. Do not list them as bullets here.

---

### SECTION 6 — Connection to Previous Session (no heading)

Write **1–2 paragraphs** that:

- Open with: "In the **previous session**," and describe what was covered immediately before this one (drawn from the course map).
- Explicitly show how the previous session's knowledge becomes a building block for this session.
- Do not be generic ("as you have been learning so far"). Be specific about the exact skills and tools.

---

### SECTION 7 — Discovery Preview

```markdown
In this pre-read, you will discover:
```

Write **exactly 4 bullet points** using this format:
```
- How to **[verb]** [what] — drawn directly from SESSION_LOs or the session objective.
```

Use verbs from this set: `understand`, `learn`, `discover`, `apply`, `connect`, `interpret`, `recognise`, `build`.

---

### SECTION 8 — Horizontal Rule

```markdown
---
```

---

### SECTION 9 — Two or Three Deep-Dive Sections

Write **2 to 3 standalone sections**, each with its own `##` heading. These are the substantive educational sections.

**How to choose the headings:**
- Each heading should correspond to a distinct conceptual challenge or insight from `SESSION_TOPICS`.
- Use question-style or statement-style headings that a student would find intriguing.
- Examples: `## Why Array Shape Is Not Just a Detail`, `## How Matrix Multiplication Connects to Prediction`, `## Where These Operations Appear in Real Life`

**How to write each section:**
- 2–4 paragraphs.
- Start with the problem or question. Explain the concept. Ground it in a concrete example or analogy. Close with a connection to either the course or a real-world application.
- Use **bold** to highlight key terms.
- Do not use bullet points inside these sections.

**Required pattern**: The last deep-dive section must always be a "Where [Topic] Appears in Real Life" section. It must cover **3–5 distinct real-world industries or use cases** where this session's skills apply, written in flowing prose (not a bulleted list of industries).

---

### SECTION 10 — What's Next

```markdown
## What's Next
```

Write 1 short introductory line: "After this session, you will be able to:"

Then write **4–6 bullet points** describing concrete, measurable skills. Each should start with a verb in the active present tense:
```
- Reshape a NumPy array from one shape to another using `reshape()` and understand when this is safe.
- Transpose a matrix using `.T` and understand how rows and columns swap.
```

Close with 2 sentences (not a bullet) that:
1. Lower the stakes: "You do not need to [X] right now."
2. Restate the mental model as a memorable phrase: "The goal is [powerful one-liner summary]."

---

### SECTION 11 — Interesting Questions for the Live Session

```markdown
## Interesting Questions for the Live Session
```

Write **exactly 4 questions** that:
- Are genuinely thought-provoking, not softballs.
- Cannot be answered with a simple yes or no.
- Push the student to think beyond definitions — towards edge cases, tradeoffs, or implications.
- Are directly relevant to `SESSION_TOPICS` and the deep-dive content of the pre-read.

Each question should be 1–2 sentences. No sub-bullets. No numbering — just 4 standalone question lines, each starting with `-`.

Close with a 1-sentence synthesis statement (not a bullet) in the format:
```
By the end of this session, [topic] should feel less like [something abstract] and more like [something practical and powerful]: **[memorable one-liner].**
```

---

## STEP 5 — SAVE THE FILE

Save the completed pre-read to a **course-specific subfolder** inside `Pre_read/`:
```
Pre_read/{Course_Folder}/{Preread_Session_Title_Slug}.md
```

**Course folder rules (`{Course_Folder}`):**
- Derive the folder name directly from `COURSE_NAME` as provided by the user.
- Lowercase the entire string.
- Replace all spaces with `-`.
- Remove any characters that are not alphanumeric, `-`, or `_`.
- If the folder already exists, use it as-is. If it does not exist, create it.

**Examples of course folder names:**
```
COURSE_NAME: "iitreict-dsai-2603"  → Folder: Pre_read/iitreict-dsai-2603/
COURSE_NAME: "iitp-aiml-2601"      → Folder: Pre_read/iitp-aiml-2601/
COURSE_NAME: "iitp-aiml-2510"      → Folder: Pre_read/iitp-aiml-2510/
COURSE_NAME: "iitp-aiml-2604"      → Folder: Pre_read/iitp-aiml-2604/
```

**Slug rules for the filename (`{Preread_Session_Title_Slug}`):**
- Take `SESSION_TITLE`.
- Replace `&`, `+`, `/` with `_and_`.
- Replace all spaces with `_`.
- Remove all other special characters (`(`, `)`, `:`, `,`, `-`).
- Use title casing — preserve capitalisation from the session title.
- Do NOT add a session number prefix (e.g., do not write `5_3_`).

**Full path examples:**
```
COURSE_NAME   : iitreict-dsai-2603
SESSION_TITLE : Numeric Manipulation & Linear Algebra Basics
→ Path: Pre_read/iitreict-dsai-2603/Preread_Numeric_Manipulation_and_Linear_Algebra_Basics.md

COURSE_NAME   : iitp-aiml-2601
SESSION_TITLE : Linear Algebra for ML
→ Path: Pre_read/iitp-aiml-2601/Preread_Linear_Algebra_for_ML.md

COURSE_NAME   : iitp-aiml-2510
SESSION_TITLE : Clustering in Practice
→ Path: Pre_read/iitp-aiml-2510/Preread_Clustering_in_Practice.md
```

---

## STEP 6 — QUALITY CHECKLIST

Before finalising the file, verify every item below. Fix anything that does not pass.

### Content Quality
- [ ] Mermaid diagram renders correctly — no raw `&`, no `style=` inside labels, no session numbers
- [ ] All previous/upcoming modules in the diagram are sourced from the actual CSV, not invented
- [ ] The current session box clearly states the mental shift or live lecture value
- [ ] No section is missing — all 11 sections from Step 4 are present
- [ ] Opening narrative does NOT start with "In today's world", "In the age of AI", or similar clichés
- [ ] The "What If" section opens with the literal phrase "What if"
- [ ] The "In this pre-read, you will discover" section has exactly 4 bullets
- [ ] The "What's Next" section ends with a non-bullet closing sentence and memorable one-liner
- [ ] The closing synthesis sentence is present at the end of Section 11
- [ ] No bullet points appear in Sections 3, 4, 5, 6 (narrative sections)

### Accuracy
- [ ] `SESSION_TITLE` matches the title in the CSV exactly (or was confirmed by the user)
- [ ] Module names in the Mermaid diagram match the CSV `Module` column verbatim (minus week ranges if space is tight)
- [ ] Tech stacks cited in the diagram appear in the actual `Topics Covered` column for those sessions
- [ ] The previous session referenced in Section 6 is the actual immediately preceding session in the CSV

### File Output
- [ ] File is saved inside `Pre_read/{course-folder}/` — the subfolder name matches the COURSE_NAME slug
- [ ] The course subfolder was created if it did not already exist
- [ ] Filename follows the slug rules from Step 5
- [ ] File uses `.md` extension
- [ ] File opens with `# Pre-read: {SESSION_TITLE}` as the first line

---

## REFERENCE — File Layout

```
pre-reads/
├── instruction.md                          ← This file (do not modify)
├── Course_Syllabus/
│   ├── iitreict-dsai-2603-...csv           ← Schema A — Module + Title + Objective
│   ├── iitreict-dsai-2605-...csv           ← Schema A — Module + Title + Objective
│   ├── IITP-AIML-2601-...csv              ← Schema B — Module + Session Title + Learning Objective
│   ├── iitp-aiml-2510-...csv              ← Schema C — Trimester + Session Title + Objective
│   ├── iitp-aiml-2604-...csv              ← Schema D — Numeric Module + Title + LO/s
│   └── [future courses added here]
├── Content_Template/
│   ├── Preread_template.md                 ← Gold-standard example pre-read (RAG session)
│   ├── Mental_Map_Template.md              ← Mermaid diagram reference example
│   └── Mind_Map_Mental_Model_Per_Session_Prompt.md  ← Prompt for diagram generation only
└── Pre_read/
    ├── iitreict-dsai-2603/                 ← One subfolder per course (auto-created)
    │   ├── Preread_Numeric_Manipulation_and_Linear_Algebra_Basics.md
    │   └── Preread_AB_Testing_and_Correlation.md
    ├── iitreict-dsai-2605/
    │   └── Preread_Classification_Evaluation_Metrics.md
    ├── iitp-aiml-2601/
    │   └── Preread_Linear_Algebra_for_ML.md
    ├── iitp-aiml-2510/
    │   └── Preread_Clustering_in_Practice.md
    ├── iitp-aiml-2604/
    │   └── Preread_Git_and_Version_Control.md
    └── [one subfolder per course, created on first use]
```

### How to Add a New Course

To support a new course in the future:
1. Add its CSV file to `Course_Syllabus/` with the course/cohort identifier in the filename.
2. The schema does **not** need to match existing files exactly. Step 1.2 handles multi-schema detection automatically for known patterns.
3. If the new CSV uses a column naming pattern not yet listed in the Known Schemas table, add a new row to that table in Step 1.2 describing the mapping.
4. No other changes to this `instruction.md` are needed — the course will be auto-discoverable via Step 1.1.

---

## REFERENCE — Mermaid Colour Palette Quick Reference

| Box type | Fill | Stroke | Usage |
|---|---|---|---|
| `previous` | `#eef4ff` | `#4f7ccf` | Previous modules + current module until prev session |
| `current` | `#fff4d6` | `#d9a321` | Current session only |
| `value` | `#eaf8ef` | `#3f9f63` | Course value + Real-life value |
| `future` | `#f4ecff` | `#8a61d1` | Upcoming modules |

All text uses `color:#111827` (near-black) for readability. Never use dark background fills.

---

## REFERENCE — Example Session Inputs and Expected Output

### Example 1 — Schema A (iitreict-dsai-2603)
```
COURSE_NAME   : iitreict-dsai-2603
SESSION_TITLE : Numeric Manipulation & Linear Algebra Basics
SESSION_LOs   : (use CSV Objective column)
SESSION_TOPICS: (use CSV Topics Covered column)
```
→ Schema detected: A (Module carry-forward, Title, Objective)
→ Output file: `Pre_read/Preread_Numeric_Manipulation_and_Linear_Algebra_Basics.md`
→ CSV row: Module 2, Week 5, Session 5.3

### Example 2 — Schema A (iitreict-dsai-2605)
```
COURSE_NAME   : iitreict-dsai-2605
SESSION_TITLE : Classification Evaluation Metrics
SESSION_LOs   : Learn to measure model performance beyond simple accuracy
SESSION_TOPICS: Confusion Matrix; Precision and Recall; F1-Score; ROC-AUC curves
```
→ Schema detected: A (Module carry-forward, Title, Objective)
→ Output file: `Pre_read/Preread_Classification_Evaluation_Metrics.md`
→ CSV row: Module 5, Week 18, Session 18.2

### Example 3 — Schema B (iitp-aiml-2601)
```
COURSE_NAME   : iitp-aiml-2601
SESSION_TITLE : Linear Algebra for ML
SESSION_LOs   : (use CSV Learning Objective column)
SESSION_TOPICS: (use CSV Topics Covered (2-hr Content Scope) column)
```
→ Schema detected: B (Module on every row, Session Title, Learning Objective)
→ GROUP_NAME = "Module 1: Foundations of AI and Data Science"
→ Output file: `Pre_read/Preread_Linear_Algebra_for_ML.md`
→ CSV row: Module 1, Week 6, Session 6.1

### Example 4 — Schema C (iitp-aiml-2510)
```
COURSE_NAME   : iitp-aiml-2510
SESSION_TITLE : Clustering in Practice
SESSION_LOs   : (use CSV Objective column)
SESSION_TOPICS: (use CSV Topics Covered column)
```
→ Schema detected: C (Trimester grouping, Session Title, Objective)
→ GROUP_NAME = "Trimester 2"
→ Note: Trimester is the grouping unit. In the Mermaid diagram, treat each Trimester as a
  "module". Show Trimester 2 as the current group, Trimester 3 as upcoming.
→ Output file: `Pre_read/Preread_Clustering_in_Practice.md`
→ CSV row: Trimester 2, Week 22, Session 22.1

### Example 5 — Schema D (iitp-aiml-2604)
```
COURSE_NAME   : iitp-aiml-2604
SESSION_TITLE : Git & Version Control
SESSION_LOs   : (use CSV Learning Objective/s column; may be empty — fall back to Topics Covered)
SESSION_TOPICS: (use CSV Topics Covered column)
```
→ Schema detected: D (Numeric Module col, Title, LO/s)
→ GROUP_NAME constructed as "Module 1" (from numeric value `1`)
→ Output file: `Pre_read/Preread_Git_and_Version_Control.md`
→ CSV row: Module 1, Week 2, Session 2.1

---

*This instruction file is designed to be passed directly as a system prompt or agent instruction. Every step is executable. No creative interpretation is needed — follow the steps and the pre-read will be consistent, accurate, and high-quality every time.*
