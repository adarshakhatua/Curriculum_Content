# Pre-read: Model Compression & Efficiency

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1(["Previous Module<br/><b>Module 1</b><br/><i>(Python, NumPy, Scikit-learn)</i><br/>Programming, data wrangling, and ML foundations"])
    C1([[Current Module Until Previous Session<br/><b>Module 2</b><br/><i>(Scikit-learn, XGBoost, PCA)</i><br/>Supervised classifiers, clustering, dimensionality reduction, optimisation]])
  end

  CS{{Current Session<br/><b>Model Compression &amp; Efficiency</b><br/><i>Think small, deploy big</i><br/>Pruning · Distillation · Quantisation · Benchmarking}})

  subgraph Value["Why This Matters"]
    CV(["Course Value<br/><b>Enables efficient deep learning deployment</b><br/>Mastering compression prepares you to deploy the neural networks and LLMs built in Module 3 at production scale."])
    RV(["Real-Life Value<br/><b>Ship AI to phones and edge devices</b><br/>Companies use these techniques to run large models on consumer hardware and serve millions of predictions at minimal cloud cost."])
  end

  subgraph Future["Where This Leads Next"]
    F1(["Upcoming Module<br/><b>Module 3</b><br/><i>(PyTorch, Transformers, LangChain)</i><br/>Deep learning, NLP, LLMs, and production AI systems"])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| C1
  C1 ==>|&nbsp;Compress&nbsp;| CS
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
  linkStyle default stroke-width:3px;
```

You train a high-accuracy image classifier and push it to production. It works perfectly in your dev environment. But when your team tries to deploy it on the company's mobile app, the model is 500 MB — too large to ship over cellular data, and it takes three seconds per inference on a phone. Your product manager tells you the app must run entirely on-device with inference under 100 milliseconds. The model you spent weeks perfecting simply does not fit.

This is not an edge case — it is the daily reality of production machine learning. Larger models deliver better accuracy, but they also consume more memory, increase latency, and drive up cloud serving costs. A single large transformer model can cost thousands of dollars per day to serve at scale. The naive approach of "just use the biggest model" fails when real-world constraints like battery life, bandwidth, and latency matter. You need the accuracy of a large model inside the body of a small one.

That is where **Model Compression & Efficiency** becomes essential. By learning to prune unnecessary weights, distill knowledge from large models into smaller ones, and represent weights with fewer bits, you can shrink a model by 10–100× while retaining most of its accuracy. These techniques are what make modern AI — from smartphones to edge devices to real-time APIs — actually feasible in production.

What if you could take a 7-billion-parameter language model and fit it onto a single consumer GPU, or deploy a real-time object detection system on a drone's onboard computer with no cloud connection? Techniques like **pruning**, **knowledge distillation**, and **quantisation** turn these scenarios from pipe dreams into engineering reality. This session gives you the toolkit to make models smaller, faster, and cheaper — without rebuilding them from scratch.

Model compression is the art of making a trained neural network smaller and faster without catastrophic accuracy loss. The core idea is deceptively simple: most deep learning models are massively overparameterised — they contain far more weights than they actually need. **Pruning** exploits this by identifying and removing low-importance weights, turning dense networks into sparse ones that compute faster. **Knowledge distillation** takes a different approach: a large, accurate "teacher" model trains a compact "student" model to mimic its outputs, transferring the teacher's knowledge into a much smaller architecture. **Quantisation** shrinks models at the bit level by representing weights with fewer bits — replacing 32-bit floating-point numbers with 8-bit or even 4-bit integers, dramatically reducing memory footprint and accelerating computation on specialised hardware. Think of it like packing for a trip. A novice packs everything they own — huge suitcase, heavy, inefficient. An experienced traveller prunes away the unnecessary items, packs only what adds real value, uses vacuum bags to compress the rest, and learns from a seasoned traveller which items earn their place. The destination is the same — accurate predictions — but the luggage is light enough to carry anywhere. In this session, you will explore how each of these techniques works under the hood, how to combine them for maximum effect, and how to benchmark the tradeoffs between model size, latency, and accuracy for your own deployment scenarios.

In the **previous session**, you explored **Advanced Optimisers & Training Techniques** — SGD with momentum, Adam, gradient clipping, and mixed precision training. Those techniques focused on training models faster and more stably, squeezing every drop of performance out of the training process. Model compression flips the perspective: instead of optimising how you train, you optimise what you have already built. The training intuition you gained — understanding learning rate dynamics, gradient behaviour, and precision formats — now serves as the foundation for making production models lean and fast. Where mixed precision used FP16 to speed up training, quantisation takes this further to INT8 and INT4 for deployment. Where gradient clipping prevented training instabilities, pruning removes entire weights to prevent deployment bloat.

In this pre-read, you will discover:

- How to **understand** why model size is a critical constraint in production ML deployments
- How to **apply** pruning to remove low-importance weights from a trained network
- How to **learn** the knowledge distillation technique for transferring a teacher's capabilities to a compact student
- How to **discover** how INT8 and INT4 quantisation reduces memory footprint while maintaining accuracy

---

## Why a 500 MB Model Fails in Production

Model size is not just a storage problem — it cascades into every aspect of deployment. A large model takes longer to load from disk, consumes more RAM during inference, and increases the latency of each prediction because the processor must move more data through memory. On a mobile device, a 500 MB model may exceed the app's total download budget or cause out-of-memory crashes on older phones. On a cloud server, larger models require more GPU memory per instance, meaning fewer concurrent users per dollar spent. The tradeoff is brutal: a 2× improvement in accuracy can lead to a 50× increase in deployment cost. In production, the model that never ships is worse than the model that ships with 98% of the accuracy. This is why compression is not an afterthought — it is a first-class design constraint that determines whether your work ever reaches a real user.

Companies like Apple, Google, and Meta have internal teams dedicated solely to model compression because they understand that a model running on a billion devices must be ruthlessly efficient. The techniques you will learn in this session — pruning, distillation, and quantisation — are the same ones that power on-device AI features like real-time photo enhancement, speech recognition, and live language translation on your smartphone.

## Three Ways to Shrink a Model — and When to Use Each

Pruning, distillation, and quantisation attack the size problem from different angles, and the best results often come from combining them. **Pruning** works by evaluating every weight in the network and scoring its importance — weights with small magnitudes or minimal impact on the loss function are set to zero. The network becomes sparse, and because zero values can be skipped during computation, inference speeds up proportionally. The beauty of pruning is that modern networks can often lose 50–90% of their weights before accuracy noticeably degrades. The challenge is determining the right sparsity level and recovering lost accuracy through fine-tuning.

**Knowledge distillation** takes an entirely different route. Instead of modifying the original model, you train a separate, smaller "student" model to imitate the behaviour of the large "teacher" model. The student does not just learn from the ground-truth labels — it learns from the teacher's soft probability distributions, which contain rich information about class similarities and decision boundaries. A ResNet-50 teacher can distil its knowledge into a ResNet-18 student that is 4× smaller yet achieves near-identical accuracy. This technique is especially powerful when you have a massive ensemble or a highly accurate but impractical model that you want to compress into a deployable form.

**Quantisation** compresses at the numerical level. A standard model uses 32-bit floating-point (FP32) for each weight. By converting to INT8 (8-bit integer), you cut memory usage by 75% and often achieve 2–4× speedups on modern hardware that includes integer-optimised instructions. INT4 quantisation pushes even further — 8× compression — but requires careful calibration to avoid accuracy loss. The key insight is that most neural network weights follow a bell-shaped distribution centred near zero; you only need enough precision to distinguish the important values, and INT8 provides 256 distinct levels, which is sufficient for most deployment scenarios. Post-training quantisation can be applied in minutes without retraining, making it the fastest path to a smaller model.

## Where Model Compression Appears in Real Life

Every major tech company that deploys AI at scale relies on these techniques daily. Apple's Neural Engine uses quantised models for on-device face recognition, Siri speech processing, and real-time photo editing — all running on a phone battery. Google's TensorFlow Lite and MediaPipe pipelines apply pruning and quantisation to deliver pose estimation and language models on Android devices. In autonomous vehicles, companies like Tesla and Waymo compress perception models so they can run at real-time frame rates on embedded hardware with strict power budgets, where a single uncompressed model would overwhelm the available compute. In healthcare, medical imaging startups quantise diagnostic models so they can run on portable ultrasound devices in rural clinics with no cloud connectivity. In large-scale cloud serving, companies like OpenAI and Anthropic use quantisation to reduce the GPU memory required for each LLM inference request, directly translating to lower API costs and higher throughput. Wherever a model must leave the research lab and enter the real world, compression determines whether that transition succeeds or fails.

## What's Next

After this session, you will be able to:

- Prune low-importance weights from a trained neural network using magnitude-based criteria
- Apply knowledge distillation by training a student model on soft targets from a teacher
- Quantise model weights from FP32 to INT8 and INT4 using calibration datasets
- Benchmark the tradeoffs between model size, inference latency, and accuracy
- Select the right compression strategy for different deployment constraints (mobile, edge, cloud)
- Evaluate the accuracy impact of each compression technique and set acceptable degradation thresholds

You do not need to memorise every compression algorithm right now. The goal is to see model size not as a fixed constraint, but as a design parameter you can optimise: **make it as small as it needs to be, and no smaller.**

## Interesting Questions for the Live Session

- If pruning removes weights that are "unimportant", how do you define importance — and what happens when different pruning criteria produce wildly different sparsity patterns?
- In knowledge distillation, the student can sometimes outperform the teacher on specific metrics. When and why does this happen, and would you design a distillation strategy to encourage it?
- Quantisation maps a continuous range of floating-point values into a discrete set of integers. What determines the optimal clipping range, and what accuracy loss might you see if you choose poorly?
- If you could use only one compression technique — pruning, distillation, or quantisation — for a real-time mobile deployment, which would you pick, and what tradeoffs would you be accepting?

By the end of this session, model compression should feel less like a set of workarounds and more like a principled engineering discipline: **making models smaller is not about losing accuracy — it is about removing what does not matter.**
