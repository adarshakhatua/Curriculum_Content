# Pre-read: Unsupervised Learning: K-Means Clustering

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1 (Weeks 1–8)</b><br/><i>(Python, scikit-learn)</i><br/>Python, data wrangling, statistics, ML pipelines])
    C1[[Current Module Until Previous Session<br/><b>Module 2 (Weeks 9–12)</b><br/><i>(scikit-learn, XGBoost)</i><br/>Supervised classifiers, ensemble methods, hyperparameter optimisation]]
  end

  CS{{Current Session<br/><b>Unsupervised Learning: K-Means Clustering</b><br/><i>From labels to patterns</i><br/>Centroid · assignment · update · elbow · silhouette · segmentation}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Foundation for unstructured data</b><br/>Clustering teaches pattern discovery without labels — the core skill behind embeddings and representation learning in Module 3.])
    RV([Real-Life Value<br/><b>Real-world pattern discovery</b><br/>Companies use K-Means daily for customer segmentation, document organisation, and data exploration — a must-have in any data scientist's toolkit.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3 (Weeks 20–36)</b><br/><i>(PyTorch, Transformers)</i><br/>Deep learning, NLP, CV, and generative AI])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Unsupervised&nbsp;| CS
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

Imagine you are handed a spreadsheet containing 10,000 customer records — purchase histories, browsing times, device types — with no labels, no segments, no prior categorisation. Your manager asks: "Find me the natural groups in this data." Without a target variable to predict, every supervised algorithm you have learned so far — KNN, SVM, Decision Trees — is powerless. You cannot train a classifier if there is nothing to classify. Sorting through ten thousand rows manually is impossible, and guessing cluster boundaries by intuition leads to inconsistent, unreproducible results. The challenge is not just grouping — it is discovering structure where none is explicitly given.

This is where **clustering**, the cornerstone of unsupervised learning, enters the picture. Instead of predicting a label, clustering algorithms uncover hidden patterns by measuring similarity between data points. That is where **K-Means Clustering** becomes essential.

What if you could walk into any organisation — e-commerce, healthcare, finance — and automatically discover meaningful customer segments, detect unusual transactions, or group thousands of support tickets into actionable categories, all without a single labelled example. After this session, you will be able to design and evaluate K-Means solutions for exactly those kinds of open-ended problems. The ability to find structure in unlabelled data is what separates analysts who wait for answers from data scientists who uncover them.

**Clustering** is the task of dividing a set of data points into groups — or clusters — such that points within the same cluster are more similar to each other than to points in other clusters. Think of sorting a mixed bowl of fruits by their characteristics: apples cluster with apples not because someone told you which is which, but because colour, shape, and size naturally separate them. **K-Means** is the most widely used clustering algorithm because it balances simplicity with speed. It works by initialising a fixed number of centroids (the "K"), assigning each point to the nearest centroid, then recalculating centroids as the mean of their assigned points — repeating until the groups stabilise. In this session you will explore the full K-Means workflow: **centroid initialisation**, the **assignment** step based on distance, and the **update** step that refines cluster boundaries. You will also learn two critical evaluation techniques — the **elbow method** for choosing K and the **silhouette score** for assessing cluster quality — and see how these tools translate to real applications like **customer segmentation** and **document grouping**.

In the **previous session**, you mastered hyperparameter tuning using GridSearchCV, RandomizedSearchCV, and Optuna — systematically searching for the best configuration of supervised models like Random Forests and XGBoost. That workflow assumed you had labelled data to validate against. Now, K-Means asks you to step into a different world entirely: no labels, no validation set, no accuracy score to optimise. The same rigorous mindset you applied to tuning — questioning assumptions, iterating, evaluating — now turns inward as you learn to judge cluster quality using structural metrics like inertia and silhouette score rather than ground-truth accuracy.

In this pre-read, you will discover:
- How to **understand** the intuition behind clustering and why unsupervised learning matters in real-world data science.
- How to **apply** the K-Means algorithm step by step — from centroid initialisation to iterative refinement.
- How to **interpret** the elbow method and silhouette score to choose the optimal number of clusters.
- How to **connect** clustering techniques to practical applications like customer segmentation and document grouping.

---

## Why "K" Is the Most Important Number You Will Choose

The K in K-Means stands for the number of clusters you expect to find. But here is the catch: in unsupervised learning, you rarely know this number in advance. Choose K too small, and vastly different groups get forcibly merged — shoppers who buy luxury watches and bargain hunters become one muddy segment. Choose K too large, and natural clusters splinter into meaningless fragments. The **elbow method** addresses this by plotting the within-cluster sum of squares (WCSS) against increasing values of K. The "elbow" — the point where the WCSS stops dropping sharply — suggests a natural K. However, elbows are not always obvious, which is why the **silhouette score** provides a second lens. It measures how similar a point is to its own cluster compared to neighbouring clusters, producing a score between −1 and 1. A high average silhouette score indicates well-separated, compact clusters. Together, these two techniques give you a principled way to choose K — not by guessing, but by listening to what the data is telling you.

## How Distance Shapes a Cluster

At its heart, K-Means is a distance-based algorithm. The assignment step computes the distance between every point and every centroid, then assigns each point to the nearest one. The most common distance metric is **Euclidean distance** — the straight-line distance you remember from geometry — but the choice of distance matters enormously. If your features are on different scales (income in thousands versus age in years), the larger scale dominates the distance calculation, silently biasing your clusters. This is why **feature scaling** — using StandardScaler or MinMaxScaler — is a non-negotiable preprocessing step before running K-Means. Once distances are computed, the update step recalculates each centroid as the arithmetic mean of all points assigned to it. The algorithm converges when assignments stop changing, usually within a few dozen iterations. Understanding this dance between assignment and update reveals why K-Means is fast, scalable, and sensitive to initial centroid placement — which is why running it multiple times with different random seeds is a standard best practice.

## Where Clustering Appears in Real Life

The moment you stop assuming you have labelled data, clustering becomes one of the most versatile tools in your arsenal. In **e-commerce**, K-Means is the engine behind customer segmentation — dividing users into high-value, bargain-seeking, or dormant groups so that marketing campaigns target each segment differently. In **healthcare**, clustering groups patients by symptom profiles or lab results, helping clinicians identify subtypes of diseases that might respond to different treatments. **Document grouping** — also called topic modelling in its more advanced forms — uses clustering to organise news articles, research papers, or support tickets into thematic buckets without anyone having to read every document. Banks apply clustering to **anomaly detection** in transaction data: rare transactions that fall far from any cluster centroid are flagged for investigation. Even **image compression** leverages K-Means: by reducing the number of colours in an image to K representative colours (cluster centroids), the image can be stored with significantly fewer bits while retaining visual fidelity. In every case, the underlying pattern is the same — discovering natural structure where no explicit labels exist.

## What's Next

After this session, you will be able to:
- Implement the K-Means algorithm from scratch and with scikit-learn's `KMeans` class.
- Choose the optimal number of clusters using the elbow method and silhouette score.
- Preprocess data appropriately — scaling features — before applying clustering.
- Evaluate cluster quality using inertia, silhouette score, and visual inspection.
- Apply K-Means to real-world problems like customer segmentation and document grouping.

You do not need to memorise every mathematical derivation right now. The goal is to see that unsupervised learning is not a mystery — it is a structured way of listening to what your data is telling you: **Clustering turns unlabelled data into discoverable structure.**

## Interesting Questions for the Live Session

- If K-Means depends heavily on initial centroid placement, how can you be confident your clusters are the "true" structure rather than an artefact of where centroids happened to land?
- The elbow method often produces a smooth curve with no clear elbow — what does that ambiguity tell you about the data, and how would you proceed?
- Silhouette score can be computed for any clustering, but it always rewards compact, spherical clusters — what types of real-world clusters would it systematically undervalue?
- K-Means uses Euclidean distance by default, but Manhattan distance and cosine similarity are also common choices — what properties of your data would guide you toward one distance metric over another?

By the end of this session, clustering should feel less like an abstract algorithm and more like a practical lens for finding hidden structure: **Unsupervised learning is not about guessing — it is about discovering.**
