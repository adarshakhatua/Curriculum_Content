# Pre-read: Dimensionality Reduction: PCA & t-SNE

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1</b><br/><i>(Python, Scikit-learn)</i><br/>Python, data analysis, and ML fundamentals])
    C1[[Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(Scikit-learn, XGBoost)</i><br/>Classifiers, ensembles, and clustering algorithms]]
  end

  CS({{Current Session<br/><b>Dimensionality Reduction: PCA &amp; t-SNE</b><br/><i>Seeing hidden structure in high dimensions</i><br/>Curse of dimensionality &middot; PCA &middot; t-SNE &middot; UMAP &middot; Explained variance}})

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Essential bridge to deep learning</b><br/>PCA and t-SNE teach you to think in feature spaces and latent representations — a skill you will use constantly in neural networks and embedding layers.])
    RV([Real-Life Value<br/><b>Visualise and compress real-world data</b><br/>From genomics to e-commerce, dimensionality reduction helps you find patterns, reduce noise, and speed up models by focusing on what matters.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3</b><br/><i>(PyTorch, HuggingFace)</i><br/>Deep learning, NLP, and production AI systems])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Reduce Dimensions&nbsp;| CS
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

You load a customer dataset with 500 behavioural, demographic, and transactional features. Your KNN classifier — which worked beautifully on the 10-feature practice data — now barely outperforms random guessing. Your K-Means clusters look like a shapeless cloud with no interpretable boundaries. You add more data, but the problem only gets worse.

This is the **curse of dimensionality** in action. As the number of features increases, the volume of the feature space expands so rapidly that your data becomes sparse — every point seems equally far from every other point. Distance metrics like Euclidean distance, which power KNN, K-Means, and DBSCAN, lose their meaning. Meanwhile, redundant features add noise, increase overfitting risk, and slow down training. More features do not always bring more signal — often they bring mostly confusion.

The solution is not to collect more rows, but to reduce the number of columns intelligently — to find the few directions that capture the real structure in your data. That is where **dimensionality reduction** becomes essential.

What if you could take a dataset with hundreds of correlated features and compress it down to just two or three dimensions — without losing the patterns that matter? You could visualise customer segments that were invisible before, speed up model training by an order of magnitude, and build simpler, more interpretable systems that generalise better. You would finally see the shape of your data instead of guessing. That is the transformation that dimensionality reduction delivers.

At its core, dimensionality reduction is about separating signal from noise. The **curse of dimensionality** describes how high-dimensional spaces are mostly empty, making every data point seem isolated and every distance calculation unreliable. The fix is to project your data into a lower-dimensional space while preserving as much meaningful structure as possible. Think of it like compressing a high-resolution photograph — you cannot keep every pixel, but you can keep enough to recognise the subject. In this session, you will explore three powerful techniques for this compression: **PCA** (Principal Component Analysis), which finds the axes of maximum variance; **t-SNE**, which excels at creating intuitive 2D visualisations of high-dimensional structure; and **UMAP**, a faster alternative that better preserves global relationships.

In the **previous session**, you explored Hierarchical and Density-Based Clustering — learning how to group data using dendrograms and DBSCAN without specifying the number of clusters upfront. Those algorithms, like all distance-based methods, rely on the assumption that distances in the feature space carry meaning. The curse of dimensionality breaks that assumption: in high dimensions, every point is roughly equidistant from every other, and clusters dissolve into noise. Dimensionality reduction directly addresses this limitation. By compressing your features into a smaller set of informative axes, you restore the power of distance-based algorithms and unlock clearer, more reliable groupings.

In this pre-read, you will discover:
- How to **recognise** the curse of dimensionality and why it makes distance-based algorithms unreliable.
- How to **apply** PCA to compress hundreds of features into a few principal components while preserving maximum variance.
- How to **use** t-SNE to create intuitive 2D visualisations that reveal hidden structure in your data.
- How to **choose** between PCA, t-SNE, and UMAP depending on whether your goal is compression, visualisation, or preserving global structure.

---

## Why More Features Can Backfire — The Curse of Dimensionality

Imagine marking ten random points along a one-metre line. The average distance between any two points is small — they are packed close together. Now imagine placing those same ten points in a square room, then in a cube, and then in a 100-dimensional hypercube. In one dimension, ten points cover the line comfortably. In 100 dimensions, those same ten points are lost in a volume so vast that every point is effectively isolated. This is the curse in a nutshell: the volume of the feature space grows exponentially with the number of dimensions, and your data becomes sparse almost instantly.

The practical consequence is devastating for machine learning. Distance metrics like Euclidean distance become nearly constant — the ratio between the nearest and farthest point converges to 1. Any algorithm that relies on proximity (KNN, K-Means, DBSCAN, SVM with RBF kernel) degrades rapidly. Model training becomes slower, overfitting becomes harder to control, and visualisation becomes impossible beyond three dimensions. You cannot simply plot 500 features to check your data quality.

Dimensionality reduction breaks this curse by projecting your data into a lower-dimensional space that preserves the relationships you actually care about. It is not about throwing away information — it is about identifying which dimensions carry signal and which carry noise, then keeping only the signal.

## PCA: Finding the Axes That Actually Matter

Principal Component Analysis solves a simple but powerful problem: given a cloud of points in high-dimensional space, which direction captures the most variation? That direction is your first **principal component**. The second component is the direction orthogonal to the first that captures the most remaining variance, and so on. PCA effectively rotates your coordinate system so that the first few axes contain most of the information, and the remaining axes contain mostly noise.

The **explained variance ratio** tells you exactly how much of the total data variance each component captures. If the first two components explain 80% of the variance, you can safely reduce 500 features to 2 dimensions without losing four-fifths of the signal. A **scree plot** — a bar chart of explained variance per component — helps you decide where to cut off: look for the "elbow" where the bars flatten out. Components after the elbow add little value and can be discarded.

PCA is deterministic, fast, and preserves global distances — making it ideal for compression, denoising, and as a preprocessing step before other algorithms. Its limitation is that it can only capture linear relationships. For data with curved or folded structures, PCA may miss the real patterns that non-linear methods like t-SNE and UMAP can reveal.

## Where Dimensionality Reduction Appears in Real Life

In **genomics and bioinformatics**, gene expression datasets routinely contain measurements for 20,000+ genes across a few hundred patient samples. PCA and t-SNE are standard tools for visualising cell types, identifying disease subtypes, and reducing the feature space before training classifiers. When researchers publish a UMAP plot of single-cell RNA-seq data, they are showing you the hidden structure of cellular diversity compressed into two dimensions.

In **e-commerce and recommendation systems**, the user-item interaction matrix is one of the highest-dimensional objects in industry — millions of users by millions of products. Matrix factorisation techniques like SVD (which is PCA for sparse matrices) decompose this sparse, high-dimensional matrix into compact user and item embeddings, enabling personalised recommendations at scale. Amazon, Netflix, and Spotify all rely on this principle.

In **natural language processing**, documents represented as TF-IDF vectors routinely have 50,000+ dimensions (one per unique word). Dimensionality reduction via TruncatedSVD — often called Latent Semantic Analysis — compresses these unwieldy vectors into dense semantic spaces where words with similar meanings cluster together, making downstream classification and retrieval far more tractable.

In **finance**, portfolio managers face hundreds of correlated asset returns. PCA identifies the few independent risk factors driving the market — typically interest rate risk, equity risk, and volatility — allowing analysts to build simpler, more robust risk models and detect anomalous trading behaviour.

In **computer vision**, raw images with 256 by 256 pixels translate to 196,608 dimensions when you account for RGB channels. PCA (historically used in eigenfaces for facial recognition) and modern learned embeddings compress visual data into compact representations, enabling fast retrieval and efficient storage of millions of images.

## What's Next

After this session, you will be able to:
- Explain the curse of dimensionality and diagnose when it is degrading your model's performance.
- Apply PCA to a dataset using Scikit-learn and interpret the explained variance ratio.
- Read a scree plot to decide how many principal components to retain for a given task.
- Use t-SNE to create insightful 2D visualisations of high-dimensional data.
- Compare PCA, t-SNE, and UMAP to select the right technique based on data size, structure, and goal.
- Preprocess high-dimensional data before feeding it into clustering or classification pipelines.

You do not need to memorise the eigendecomposition behind PCA right now. The goal is to develop an intuition for when and why to reduce dimensions — because fewer features, chosen wisely, almost always lead to better models.

## Interesting Questions for the Live Session

- If two features in your dataset are perfectly correlated, PCA will collapse them into a single component — but does that mean you should always remove correlated features before PCA, or can PCA handle multicollinearity on its own?
- t-SNE is known to produce visually convincing clusters even on random noise — what practical checks can you apply to ensure the structure you see in a t-SNE plot is real and not an artefact of the algorithm?
- PCA maximises variance, but the directions of maximum variance are not always the directions most useful for a classification task — how might you adapt your approach if your goal is to separate classes rather than preserve variance?
- UMAP preserves more global structure than t-SNE while being significantly faster — if you had to analyse a dataset with 100,000 samples and 1,000 features, which tradeoffs would drive your choice between the two methods?

By the end of this session, dimensionality reduction should feel less like a black-box compression trick and more like a principled way to reveal the true shape of your data: **Fewer dimensions don't mean less information — they mean signal without noise.**
