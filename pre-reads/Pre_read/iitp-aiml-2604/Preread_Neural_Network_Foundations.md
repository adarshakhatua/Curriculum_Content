# Pre-read: Neural Network Foundations

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1(["Previous Module<br/><b>Module 1</b><br/><i>(Python, NumPy, Pandas, Scikit-learn)</i><br/>Data programming, statistics, and ML basics"])
    C1[["Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(Scikit-learn, XGBoost, PCA)</i><br/>Classifiers, clustering, dimensionality reduction, optimisation"]]
  end

  CS({{Current Session<br/><b>Neural Network Foundations</b><br/><i>From logistic regression to a neuron</i><br/>Perceptron · MLP forward pass · activation functions}})

  subgraph Value["Why This Matters"]
    CV(["Course Value<br/><b>Gateway to all deep learning</b><br/>This session unlocks CNNs, RNNs, Transformers, and LLMs — every major architecture ahead builds on the neuron."])
    RV(["Real-Life Value<br/><b>Foundation for industry AI</b><br/>Neural networks power image recognition, chatbots, and recommendation engines across every industry."])
  end

  subgraph Future["Where This Leads Next"]
    F1(["Upcoming Module<br/><b>Module 3</b><br/><i>(PyTorch, HuggingFace Transformers)</i><br/>Deep learning, NLP, LLMs, and production AI"])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Neural Step&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px;
```

You submit your latest classification model — a tuned XGBoost with early stopping — and the accuracy holds at 72%. The data is structured, the pipeline is clean, but the ceiling will not budge. You know the model is powerful, yet raw pixels refuse to cooperate the way tabular features do. Every classical algorithm you throw at the problem demands handcrafted features: colour histograms, edge densities, texture descriptors. Each new feature buys you a fraction of a percent. You are doing the learning, not the model.

The deeper problem is one of representation. Logistic regression, SVMs, and random forests are exceptional decision-makers, but they cannot discover features on their own. They depend entirely on what you feed them. When the input space is high-dimensional and unstructured — pixels, audio samples, word sequences — feature engineering becomes the bottleneck. No matter how well you tune the hyperparameters, the model cannot see what you did not think to extract. This is the fundamental limitation that drove an entire field to ask: what if the model could learn its own representations?

That is where **Neural Network Foundations** becomes essential. Instead of engineering features by hand, you design an architecture — a stack of simple computational units — that discovers patterns layer by layer, abstraction by abstraction, directly from raw data. This session gives you the mental model to understand how that transformation works.

What if you could build a system that learns its own features without anyone telling it what to look for? Imagine feeding raw pixel values into a model that, entirely on its own, discovers edges in the first layer, textures in the second, shapes in the third, and object parts in the fourth — all from exposure to data alone. What if the same approach could be applied to words, sounds, sensor readings, or medical scans, each time learning representations more nuanced than any human engineer could specify? This is not speculation — it is what modern neural networks already do, and understanding the foundations of how they work is the key to unlocking that capability for your own problems.

Think of a neural network as a team of medical specialists diagnosing a patient. The first-layer specialists look for basic symptoms: fever, cough, fatigue. Each one receives signals from the patient data, weighs them by what it considers important, and sends a conclusion forward only if the evidence crosses a threshold. The next-layer specialists receive those signals and combine them into syndromes: respiratory infection, inflammation. Deeper specialists combine syndromes into a diagnosis: pneumonia. Every specialist along the chain learns which signals matter and how strongly to weigh them. In a neural network, those specialists are called **neurons** (or perceptrons), the signal they receive is a **weighted sum of inputs plus a bias**, the threshold decision is an **activation function**, and the layered chain of specialists is called a **multi-layer perceptron (MLP)**. In this session, you will explore exactly how a single neuron works, how multiple neurons stack into layers, and when to use each activation function — **ReLU**, **Sigmoid**, **Tanh**, and **Softmax** — to shape what each layer learns.

In the **previous session**, you explored model compression — pruning low-importance weights, distilling large models into smaller student networks, and quantising weights for efficient deployment. All of those techniques assumed a trained model existed already and focused on making it leaner. That pragmatic lens now flips: instead of optimising an existing model, you will learn how to design the model itself. The efficiency mindset carries forward — every design choice in a neural network (number of neurons, choice of activation, depth of layers) is a tradeoff between capacity and computational cost. The same principle that drove you to question "does every weight matter?" now drives you to ask "does every neuron serve a purpose?"

In this pre-read, you will discover:

- How to **connect** logistic regression to a single neuron — and why this mental shift matters
- How to **build** the forward pass of a multi-layer perceptron step by step
- How to **recognise** when to use ReLU, Sigmoid, Tanh, or Softmax in different layers
- How to **interpret** neural networks as a natural evolution of the models you already know

---

## Why Logistic Regression Is Already a Neuron

You already know logistic regression: you feed in features, multiply each by a learned weight, add a bias, and push the result through a sigmoid function to get a probability between 0 and 1. Here is the insight that makes neural networks feel less like a leap and more like a step: a single artificial neuron does exactly the same thing. The formula is identical — **y = σ(w·x + b)** — where **σ** is the activation function, **w** is the weight vector, **x** is the input, and **b** is the bias. When that activation function is the sigmoid, the neuron and logistic regression are mathematically indistinguishable.

The only difference is vocabulary and intent. In logistic regression, you call **w·x + b** the logit (or log-odds) and treat the output as a probability. In a neural network, you call **w·x + b** the pre-activation and treat the output as the neuron's "firing" strength. A logistic regression model uses one decision boundary for the entire problem. A neural network, by stacking multiple such neurons in layers, creates many decision boundaries that work together — each neuron learns to specialise in a different pattern. The first layer might learn simple edges; the second layer learns combinations of edges; deeper layers learn increasingly abstract features.

This realisation is empowering because everything you already know about training still applies. **Gradient descent** from Session 7.1 works the same way — you compute a loss, take its derivative with respect to each weight, and nudge the weights in the direction that reduces error. **Regularisation** from Session 8.2 (L1, L2, dropout) still prevents overfitting. **Cross-validation** from Session 8.2 still measures how well your model generalises. The core mechanics of learning have not changed — only the architecture that receives them has. This continuity is why moving from classical ML to neural networks feels less like switching fields and more like upgrading your toolkit.

## The MLP Forward Pass — How Information Flows Through Layers

A **multi-layer perceptron (MLP)** organises neurons into three kinds of layers: an **input layer** that receives the raw data, one or more **hidden layers** that transform the data internally, and an **output layer** that produces the final prediction. Each neuron in a hidden or output layer is connected to every neuron in the previous layer — this is called a **fully connected** (or dense) architecture. The forward pass is the process of pushing an input through all layers to produce an output, and it happens one layer at a time.

Consider a tiny MLP with one hidden layer of three neurons and an output layer of one neuron. The forward pass unfolds as follows: the input vector **x** enters the hidden layer, where each neuron computes its own weighted sum **z₁ = w₁·x + b₁**, then applies an activation function **a₁ = f(z₁)**. The three activations from the hidden layer become the input vector for the output neuron, which computes **z₂ = w₂·a₁ + b₂** and applies a final activation — sigmoid for binary classification, softmax for multi-class, or linear for regression — to produce the prediction. Every computation is a simple linear combination followed by a non-linear squashing function, repeated layer by layer.

Activation functions are where the network gains its power to learn complex patterns. Without them, every layer would be a linear transformation, and a stack of linear layers collapses mathematically into a single linear layer — no benefit from depth. Each activation function serves a specific role. **Sigmoid** squashes values between 0 and 1, making it natural for binary classification outputs, but it suffers from vanishing gradients for very large or very small inputs. **Tanh** squashes between -1 and 1 and is zero-centred, which often helps gradients flow more stably in hidden layers. **ReLU** (Rectified Linear Unit) passes positive values through unchanged and sets negatives to zero — it is simple, computationally cheap, and the default choice for most hidden layers today because it avoids vanishing gradients. **Softmax** converts a vector of raw scores into a probability distribution (all positive, sum to 1), making it the standard for multi-class output layers. The art of choosing an activation comes down to where it sits in the network: hidden layers almost always use ReLU (or its variants), binary output layers use sigmoid, multi-class outputs use softmax, and tanh appears when you need centred activations, typically in older architectures or specific sequence models.

## Where Neural Networks Appear in Real Life

Computer vision was the first domain where neural networks demonstrated transformative impact. Systems that read medical scans for tumours, power autonomous vehicle perception pipelines, and sort billions of photos on cloud platforms all rely on convolutional neural networks — a specialised architecture that builds directly on the MLP concepts you are learning here. The neuron, the forward pass, and the activation function are the atomic units behind every modern vision system, from iPhone face unlock to satellite imagery analysis.

Natural language processing underwent a similar revolution. Every major language model — BERT for search ranking, GPT for text generation, T5 for translation — is a neural network at its core, using the same fundamental building blocks scaled to hundreds of billions of parameters. When you use a chatbot, a spell checker, or a real-time transcription service, you are relying on neural architectures where the forward pass (now transformed by attention mechanisms) still follows the same input-to-output flow you are about to study.

Beyond these headline applications, neural networks drive recommendation engines at Netflix and Spotify, fraud detection systems at every major bank, drug discovery pipelines that predict molecular properties, and energy load forecasting for smart grids. The architecture varies — MLP, CNN, Transformer, graph neural network — but the foundations are identical: weighted sums, biases, activation functions, and layered computation. The session you are about to attend is the one that makes all of those applications intelligible.

## What's Next

After this session, you will be able to:

- Explain how a single neuron performs the same computation as a logistic regression model
- Trace the forward pass through any feedforward network, layer by layer, identifying where each computation happens
- Choose between ReLU, Sigmoid, Tanh, and Softmax for different positions in a network based on the task requirements
- Visualise how information transforms as it moves from input through hidden layers to output
- Recognise when a problem is a candidate for neural networks versus classical machine learning approaches

You do not need to implement a full neural network from scratch or memorise every activation function derivative right now. The goal is to see every neural network as a stack of familiar operations — weighted sums, biases, and activation functions — working together to learn patterns you did not explicitly program.

## Interesting Questions for the Live Session

- If a single neuron is mathematically equivalent to logistic regression, what capability does adding a second hidden layer introduce that a single layer fundamentally cannot achieve?

- What would happen if you used a linear activation function — or no activation at all — in every layer of a deep network? Does depth still help, and if not, where does the collapse happen?

- ReLU is the default for hidden layers, but it suffers from "dead neurons" that never activate. How would you diagnose dead neurons in practice, and when would you choose Tanh or Leaky ReLU instead?

- Softmax guarantees outputs sum to 1, making it ideal for single-class predictions. How would you design the output layer for a multi-label problem where one input can belong to several classes simultaneously?

By the end of this session, neural networks should feel less like mysterious black boxes and more like a logical extension of the machine learning intuitions you already have: **a stack of logistic regressions learning features together.**
