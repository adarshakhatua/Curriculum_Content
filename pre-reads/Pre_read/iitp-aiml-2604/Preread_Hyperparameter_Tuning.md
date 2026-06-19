# Pre-read: Hyperparameter Tuning

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1</b><br/><i>(Python, Scikit-learn)</i><br/>Python, data wrangling, statistics, ML pipelines])
    C1[[Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(KNN, XGBoost, LightGBM)</i><br/>Distance-based, probabilistic, tree, ensemble, gradient boosting classifiers]]
  end

  CS{{Current Session<br/><b>Hyperparameter Tuning</b><br/><i>Systematic optimisation mindset</i><br/>GridSearchCV · RandomizedSearchCV · Optuna · search spaces · interpreting results}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Unlocks model performance mastery</b><br/>Mastering hyperparameter tuning lets you systematically improve every model you build from KNN to neural networks, transforming default configurations into optimised solutions.])
    RV([Real-Life Value<br/><b>Turns intuition into precision</b><br/>In production ML, tuned hyperparameters regularly boost accuracy by 5–20%, turning good-enough models into deployable, competitive systems.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3</b><br/><i>(PyTorch, HuggingFace)</i><br/>Deep learning, NLP, CV, generative AI])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Optimisation&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px
```

You spend days building a classification pipeline — cleaning data, engineering features, selecting the right algorithm. You train an XGBoost model with default settings, and it performs... okay. Your accuracy is decent, but you cannot shake the feeling that there is more left on the table. In practice, most models ship with generic defaults that work reasonably across many datasets but are rarely the best fit for yours.

The intuitive approach is to tweak a few parameters manually — raise `n_estimators` here, lower `learning_rate` there — hoping each change nudges performance upward. But this trial-and-error process is slow, inconsistent, and prone to overfitting. You might spend hours changing one parameter at a time, only to end up with a configuration that works on your validation set but fails in the wild. The number of possible combinations explodes as you add more parameters, and hand-tuning quickly becomes impossible. That is where systematic **hyperparameter tuning** becomes essential.

What if you could hand any model — a Random Forest, an XGBoost, a neural network — to an automated system that would explore thousands of parameter combinations, evaluate each with cross-validation, and return the best configuration in the time it takes you to drink your morning coffee? What if that same system could learn from past trials, focus on the most promising regions of the search space, and even restart from where it left off? This is not a fantasy — it is exactly what tools like **GridSearchCV**, **RandomizedSearchCV**, and **Optuna** do every day in production ML pipelines. This session gives you the keys to that capability.

At its core, **hyperparameter tuning** is the process of systematically searching for the best configuration of a model's settings before training. Unlike model parameters, which are learned from data during training (like the coefficients in linear regression or the split points in a decision tree), hyperparameters are the knobs you turn before training begins — things like the depth of a tree, the learning rate of a gradient booster, or the regularisation strength. Getting them right is the difference between a model that barely works and one that shines. Think of it like adjusting a high-end camera. You could leave it on auto mode (default parameters) and capture decent photos in most conditions. But when the lighting is tricky or you are shooting action shots, you need to dial in the aperture, shutter speed, and ISO manually. Each setting interacts with the others, and the perfect combination depends on your specific scene. Hyperparameter tuning is that same act of deliberate calibration — except the camera is your ML model, and the scene is your dataset.

In this session, you will explore three distinct approaches to this search: **GridSearchCV**, which exhaustively tries every combination in a predefined grid; **RandomizedSearchCV**, which samples randomly for efficiency; and **Optuna**, which uses automated optimisation to intelligently navigate the search space. You will learn how to **define a search space** using ranges and probability distributions, and how to **read and interpret tuning results** to make informed decisions about your models.

In the **previous session**, you worked with **LightGBM**, a high-performance gradient boosting framework that introduced leaf-wise tree growth, native categorical feature handling, and impressive training speed. You saw how its hyperparameters — like `num_leaves`, `min_child_samples`, and `learning_rate` — dramatically affect model behaviour. But you ran those models with manually chosen settings, relying on intuition and experience to set each knob. That approach works for a first pass, but it leaves performance on the table. Now, with **hyperparameter tuning**, you move from guessing to searching — systematically exploring the space of configurations to find the combination that makes LightGBM (and every other model you have learned since Module 2 began) perform at its peak.

In this pre-read, you will discover:

- How to **apply** GridSearchCV for exhaustive hyperparameter search with cross-validation
- How to **use** RandomizedSearchCV when the search space is too large for exhaustive search
- How to **automate** optimisation with Optuna's intelligent trial-and-error approach
- How to **interpret** tuning results and recognise when a model is being overfitted by aggressive tuning

---

## Why Exhaustive Search Is Not Always the Answer

When you first encounter **GridSearchCV**, it feels like the obvious solution: define a grid of possible values for each hyperparameter, try every combination with cross-validation, and pick the winner. For small search spaces — say, three parameters with five values each — this works beautifully. You evaluate 5 × 5 × 5 = 125 combinations, and the best configuration is guaranteed to be somewhere in that grid. The exhaustive nature of GridSearchCV gives you certainty: you have explored the entire predefined space. But that certainty comes at a cost. Add just two more parameters with five values each, and you are suddenly evaluating 5⁵ = 3,125 combinations. With 5-fold cross-validation, that is 15,625 training runs. For a model like LightGBM that takes thirty seconds per run, you are looking at over five days of computation.

This is the **curse of dimensionality** applied to search: the grid grows exponentially with each new parameter. **RandomizedSearchCV** breaks this deadlock by sampling a fixed number of random combinations from the search space. It does not guarantee finding the absolute best configuration, but research shows it often finds a near-optimal one in a fraction of the time — especially when only a few parameters truly matter. The tradeoff between thoroughness and speed is one you will navigate every time you tune a model. Knowing when to exhaust the entire grid and when to sample randomly is a skill that separates efficient practitioners from those who burn weeks of compute.

## How Optuna Thinks Differently About Search

GridSearchCV and RandomizedSearchCV share a fundamental limitation: each trial is independent. Running a bad configuration tells you only that this specific combination performs poorly — it does not inform the next choice. **Optuna** changes this by treating hyperparameter search as an optimisation problem rather than a sampling exercise. It uses algorithms like the **Tree-structured Parzen Estimator (TPE)** to build a probabilistic model of which hyperparameter values are likely to yield good results, then uses that model to suggest new configurations.

Imagine a blindfolded hiker trying to find the lowest point in a valley. GridSearchCV would place markers at every intersection on a map and check them all. RandomizedSearchCV would drop markers at random locations. Optuna, by contrast, takes a step, feels the slope, and adjusts direction based on what it has learned. It builds a **surrogate model** of the objective function (your validation score) and uses it to focus on promising regions while still exploring unfamiliar areas to avoid local optima. Optuna also supports **pruning** — automatically stopping trials that are clearly underperforming early in training, saving enormous amounts of compute. This session will give you hands-on experience defining search spaces with Optuna's `suggest_int`, `suggest_float`, and `suggest_categorical` APIs, and watching the optimisation progress in real time.

## Where Hyperparameter Tuning Appears in Real Life

Hyperparameter tuning is not an academic exercise — it is a core workflow in every serious ML organisation. In **financial services**, credit risk models are tuned to balance precision and recall at specific regulatory thresholds, where a fraction of a percent improvement translates to millions in loss avoidance. **E-commerce platforms** like Amazon and Shopify use automated tuning to optimise recommendation system parameters — the balance between exploration and exploitation in collaborative filtering, the number of latent factors in matrix factorisation — to maximise click-through and conversion rates. In **healthcare AI**, diagnostic models for medical imaging are tuned for sensitivity (recall) above all else, since missing a positive case carries higher cost than a false alarm; tuning frameworks like Optuna help clinicians navigate this tradeoff systematically. **Autonomous vehicle companies** tune perception models with hundreds of hyperparameters — from anchor box sizes in YOLO to confidence thresholds for object detection — where a well-tuned model can mean the difference between detecting a pedestrian and missing them. Even **NLP pipelines** for chatbots and virtual assistants depend on tuning the temperature, top-k, and top-p sampling parameters in large language models to balance creativity against coherence. In every case, the ability to define a search space, run an automated tuning study, and interpret the results separates teams that ship robust models from teams that stay stuck with default configurations.

## What's Next

After this session, you will be able to:

- Define a search space for any scikit-learn or gradient boosting model using ranges, lists, and probability distributions
- Run GridSearchCV with cross-validation and interpret the resulting `cv_results_` to identify the best parameter combination
- Apply RandomizedSearchCV to large search spaces and understand when random sampling beats exhaustive search
- Automate hyperparameter optimisation with Optuna, including defining objective functions and using TPE-based sampling
- Use Optuna's visualisation tools to analyse trial history, parameter importances, and convergence
- Recognise common tuning pitfalls such as overfitting to the validation set and wasteful search space design

You do not need to memorise every parameter of every model right now. The goal is to develop a systematic optimisation mindset: **defaults are starting points, not destinations**.

## Interesting Questions for the Live Session

- When you tune hyperparameters on the same validation set repeatedly, are your final metrics still trustworthy, or have you inadvertently leaked information?
- If GridSearchCV is guaranteed to find the best combination within the grid, why would any professional ever choose RandomizedSearchCV over it?
- Optuna uses past trials to guide future ones — does this create a risk of converging too quickly to a local optimum and missing a better region of the search space?
- A model tuned on your training data achieves 94% accuracy, but drops to 81% on a holdout test set. Which of the tuning tools covered in this session could help diagnose and fix this gap?

By the end of this session, hyperparameter tuning should feel less like guesswork and more like a structured engineering discipline: **systematic search beats intuition every time**.
