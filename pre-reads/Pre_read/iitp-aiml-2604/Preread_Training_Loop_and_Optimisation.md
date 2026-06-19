# Pre-read: Training Loop & Optimisation

## Context of This Session in the Course

%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1</b><br/><i>(Python &amp; Pandas, Scikit-learn &amp; Calculus)</i><br/>Python, data analysis, ML pipelines])
    C1[[Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(Scikit-learn Classifiers, PyTorch Autograd)</i><br/>Classifiers, clustering, reduction, PyTorch]]
  end

  CS{{Current Session<br/><b>Training Loop &amp; Optimisation</b><br/><i>Turning math into running code</i><br/>Loss functions · Training loop · Schedulers · Early stopping}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Unlocks all deep learning</b><br/>The training loop is the engine that powers every neural network you will build in Module 3 and beyond.])
    RV([Real-Life Value<br/><b>Train production neural networks</b><br/>Enables you to train, debug, and optimise real models for computer vision, NLP, and generative AI tasks.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3</b><br/><i>(CNNs &amp; CV, Transformers &amp; LLMs)</i><br/>Deep learning, NLP, CV, generative AI])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Train Loop&nbsp;| CS
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

You just finished building your first neural network in PyTorch — a few `nn.Linear` layers, a ReLU activation, and some carefully chosen weight initialisations. You run the training script and watch the loss number print every epoch. It drops from 4.2 to 1.8 in the first few iterations, then slows to a crawl. At epoch 20, it starts creeping back up. Should you stop? Lower the learning rate? Change the loss function? The terminal gives you numbers, but no answers.

This uncertainty is the defining challenge of moving from theory to practice in deep learning. Reading about gradient descent and understanding the math is one thing. Actually orchestrating the four-step cycle — forward pass, loss computation, backward pass, parameter update — across hundreds or thousands of iterations demands a different kind of knowledge. Without a clear mental model of the training loop, you are flying blind, relying on guesswork and luck to reach convergence.

That is where **Training Loop & Optimisation** becomes essential.

What if your manager handed you a dataset of customer reviews and asked you to train a sentiment classifier from scratch — no pre-trained models, no shortcuts — and deliver it by the end of the week? You would need to choose the right loss function, write a robust training loop, schedule the learning rate to converge reliably, and stop at the exact moment the model generalises best. After this session, that request shifts from intimidating to a straightforward engineering task.

The **training loop** is the core machinery that turns a static neural network into a learning system. It follows an unbroken four-step cycle: zero the accumulated gradients, run a **forward pass** to get predictions, compute the **loss** by comparing predictions to ground truth, execute **backpropagation** to calculate gradients for every parameter, and finally take an **optimiser step** to update those parameters. Think of it like tuning a guitar by ear. The forward pass is you plucking a string. The loss function is your ear telling you how far off the pitch is. Backpropagation traces which tuning peg (which weight) to turn. The optimiser turns that peg. A **learning rate scheduler** — such as `StepLR` or `ReduceLROnPlateau` — adjusts how aggressively you turn the peg as you get closer to the right pitch. **Early stopping** is the discipline to stop tuning once the note sounds right, before you over-tighten and snap the string. In this session, you will implement each piece: **MSELoss** and **CrossEntropyLoss** for different problem types, the full training loop in PyTorch, schedulers that adapt the learning rate dynamically, and early stopping to prevent overfitting.

In the **previous session**, you got hands-on with PyTorch Fundamentals — creating tensors, moving them to GPU, using `autograd` to compute gradients automatically, and defining layers with `nn.Module`. You learned how PyTorch tracks every operation in a computational graph and makes backpropagation a single function call. That knowledge does not disappear here — it becomes the foundation. The tensors you created are now the data flowing through your network. The `autograd` engine you practised with now runs silently inside every `.backward()` call. The `nn.Module` you assembled is now the architecture being trained. The previous session gave you the components; this session shows you the machine they power.

In this pre-read, you will discover:

- How to **build** a complete training loop using the zero_grad → forward → loss → backward → step pattern
- How to **recognise** when to use MSELoss versus CrossEntropyLoss for different tasks
- How to **apply** learning rate schedulers (StepLR, ReduceLROnPlateau) to steer model convergence
- How to **implement** early stopping to halt training at the optimal point and prevent overfitting

---

## What Makes a Training Loop Tick

Every training run in deep learning follows the same four-step dance: forward pass, loss computation, backward pass, parameter update. The pattern never changes whether you are training a linear regression model or a 7-billion-parameter language model. What changes is the scale, the data, and the architecture wrapped around this invariant core.

The first step — `zero_grad` — is easy to overlook but critical. Gradients accumulate by default in PyTorch. If you skip this call, the gradients from the previous batch add to the current one, producing a corrupted update. This is not a bug — it is by design, enabling gradient accumulation across multiple batches when memory is tight. But in a standard loop, forgetting it silently breaks your training. The forward pass then runs your input through the network to produce predictions. The loss function compares those predictions to the true labels and returns a scalar that quantifies the error. Calling `.backward()` on that scalar triggers the chain rule through every operation in the computational graph, filling `.grad` on every parameter that has `requires_grad=True`. Finally, the optimiser's `.step()` applies those gradients to update the weights — usually via SGD or Adam.

This cycle might seem mechanical, but every decision within it matters. The loss function you choose defines what "good" means. The learning rate you set determines how boldly the optimiser moves. And when you choose to stop training determines whether your model generalises or memorises. Mastering this loop means you can take any architecture, any dataset, and any task and train it with confidence.

## Choosing the Right Loss Function: Regression vs Classification

The loss function is the objective your model is trying to minimise. Pick the wrong one, and your model may never converge — or worse, converge to something useless. The two most common loss functions in PyTorch map cleanly to the two most common problem types.

**MSELoss** (Mean Squared Error) computes the average squared difference between predicted and target values. Use it when your output is a continuous number — predicting house prices, weather temperature, stock returns, or any regression task. Squaring the error means large mistakes are penalised much more heavily than small ones, which makes the model conservative about big predictions. **CrossEntropyLoss** is the standard for multi-class classification. It combines a softmax layer (which converts raw scores into a probability distribution) with negative log-likelihood (which penalises low confidence on the correct class). If your task is classifying images, detecting spam, or predicting a category from a fixed set, this is your default. One common pitfall: CrossEntropyLoss in PyTorch expects raw logits, not softmax outputs — applying softmax before the loss function double-softmaxes your predictions and breaks training.

A useful instinct when choosing: ask whether your output is bounded or unbounded, and whether it represents a quantity or a category. Regression outputs are unbounded quantities; classification outputs are probabilities over a fixed set of labels. Matching the loss to this distinction is the difference between a model that learns and one that flounders.

## Where Training Optimisation Appears in Real Life

Learning rate schedulers and early stopping are not academic niceties — they are essential tools in every production deep learning pipeline. In **autonomous vehicles**, perception models for object detection and lane segmentation are trained with cosine annealing schedulers that cycle the learning rate up and down, helping the model escape sharp local minima. Early stopping is critical here because overfitting on a limited driving dataset can produce catastrophic failures. In **healthcare AI**, medical image diagnosis models are trained with `ReduceLROnPlateau` to gracefully navigate plateaus, combined with weighted loss functions that handle severe class imbalance (far more healthy scans than diseased ones). Early stopping prevents the model from memorising noise in small, hard-to-collect medical datasets.

In **financial services**, time series models for risk assessment or algorithmic trading use `StepLR` schedulers that decay the learning rate at predetermined milestones, mirroring the intuition that coarse adjustments early in training give way to fine-tuning later. **Natural language processing** pipelines for chatbots and search engines train transformer models with a warmup-linear-decay schedule and `CrossEntropyLoss`, and early stopping is applied at the validation-perplexity checkpoint to avoid overfitting to the training corpus. **Recommendation systems** at e-commerce platforms train on billions of user interactions with custom loss functions and aggressive early stopping, because a model that overfits to popular items will fail to surface niche products that drive long-term engagement. In every case, the training loop — forward, measure, backpropagate, update — runs thousands of times per experiment, and the tools you learn in this session determine whether those runs produce a usable model or a waste of GPU hours.

## What's Next

After this session, you will be able to:

- Write a complete training loop in PyTorch following the zero_grad → forward → loss → backward → step pattern
- Select and apply MSELoss for regression problems and CrossEntropyLoss for classification tasks
- Configure StepLR and ReduceLROnPlateau learning rate schedulers to improve convergence
- Implement early stopping that monitors validation loss and restores the best model weights
- Interpret training and validation loss curves to diagnose underfitting, overfitting, and learning rate issues
- Combine schedulers and early stopping into a robust training pipeline

You do not need to memorise every scheduler parameter or loss function variant right now. The goal is to internalise the rhythm of the training loop: **forward, measure, backpropagate, update — repeat until done.**

## Interesting Questions for the Live Session

- If you forget to call `zero_grad()` before each training step, what happens to the gradients and can this ever be useful in practice?
- CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss into one operation — when would you want to decompose it and call each step separately?
- ReduceLROnPlateau reduces the learning rate when validation loss stagnates — how do you distinguish a genuine plateau from a temporary fluctuation caused by a noisy batch?
- Early stopping requires choosing a patience value — what tradeoffs are you making when you set patience to 5 versus 50, and how would you determine the right value for a given dataset?

By the end of this session, the training loop should feel less like a black-box incantation and more like a programmable assembly line: **forward, measure, backpropagate, update — repeat with purpose.**
