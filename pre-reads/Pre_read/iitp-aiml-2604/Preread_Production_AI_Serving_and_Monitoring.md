# Pre-read: Production AI: Serving & Monitoring

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1([Previous Module<br/><b>Module 1</b><br/><i>(Python, NumPy/Pandas)</i><br/>Programming and classical ML foundations])
    P2([Previous Module<br/><b>Module 2</b><br/><i>(XGBoost, PyTorch)</i><br/>ML to deep learning transition])
    C1[[Current Module Until Previous Session<br/><b>Module 3</b><br/><i>(Transformers, Diffusion Models)</i><br/>Full deep learning through generative AI]]
  end

  CS{{Current Session<br/><b>Production AI: Serving &amp; Monitoring</b><br/><i>From notebooks to production</i><br/>ML serving · vLLM · drift detection · monitoring · cost optimisation}}

  subgraph Value["Why This Matters"]
    CV([Course Value<br/><b>Unlocks production deployment skills</b><br/>This session bridges model development and real-world serving — turning trained models into deployed systems that deliver value at scale.])
    RV([Real-Life Value<br/><b>Deploy AI in production environments</b><br/>Learn to serve models efficiently, monitor them for drift, and optimise cloud costs — daily skills for any ML engineering role.])
  end

  subgraph Future["Where This Leads Next"]
    F1([Upcoming Module<br/><b>Module 3 Remaining</b><br/><i>(Vision-Language Models, Multimodal RAG)</i><br/>Multimodal AI and capstone integration])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;ML to DL&nbsp;| C1
  C1 ==>|&nbsp;Production&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px
```

Your team has just finished training a state-of-the-art LLM that achieves impressive results on your benchmark. The model card is clean, the evaluation metrics are solid, and the stakeholders are excited. Now comes the hard part: making this model actually serve predictions to thousands of users without crashing, costing a fortune, or silently degrading over time.

Deploying a model is not as simple as calling `model.predict()` on a server. Naively loading a large language model into memory for every request would exhaust your GPU budget in minutes. And even if you get it running, the data your model sees in production will drift from your training distribution — slowly at first, then all at once — turning your accurate model into a liability without anyone noticing until the dashboards turn red.

That is where **Production AI: Serving & Monitoring** becomes essential — the discipline of deploying, serving, and monitoring machine learning models in production environments where reliability, efficiency, and observability are non-negotiable.

What if you were tasked with deploying a summarisation model that needs to handle 10,000 requests per minute while keeping latency under 200 milliseconds and your cloud bill under a strict monthly budget? You would need to choose between real-time REST endpoints and batch processing, decide whether to quantise the model to fit available hardware, set up monitoring to catch data drift before it impacts users, and design a dashboard that tells you at a glance whether the system is healthy. This session equips you with exactly those decision-making frameworks.

**ML model serving** is the practice of making trained models available to other systems via APIs. The key distinction is between **batch inference** — processing large volumes of data on a schedule — and **real-time inference** — responding to individual requests with low latency. Think of batch inference like a meal-prep service that delivers weekly portions, while real-time inference is a short-order cook responding to each order as it comes in. Each pattern has different infrastructure requirements, cost profiles, and use cases. Serving large language models introduces unique challenges because a single LLM can be tens of gigabytes, far too large to reload for every request. **vLLM** addresses this with efficient memory management through PagedAttention, allowing high-throughput serving with GPU memory that would otherwise be wasted on key-value caches. **Quantisation** further compresses models by reducing the precision of weights — from 32-bit floats to 8-bit or even 4-bit integers — trading a small accuracy loss for dramatic reductions in memory footprint and latency. Once your model is serving, the work continues: **data drift** occurs when the statistical properties of incoming data change over time, while **prediction drift** signals shifts in the model's output distribution. Monitoring dashboards track these alongside latency, throughput, error rates, and memory utilisation. **Cost optimisation** strategies — right-sizing instances, using spot instances, and implementing auto-scaling — ensure your deployment remains economically viable as demand grows.

In the **previous session**, you explored Stable Diffusion and compared GAN versus diffusion model architectures, understanding how generative models create images from text prompts and how large these models are to run. That session gave you hands-on experience with the scale and resource demands of modern generative AI. Now, Production AI: Serving & Monitoring takes that understanding and asks a practical question: once you have built these powerful models, how do you make them accessible, reliable, and cost-effective in a real-world deployment where users, traffic, and data are unpredictable?

In this pre-read, you will discover:
- How to **understand** the tradeoffs between batch and real-time serving patterns for ML models.
- How to **recognise** the role of vLLM and quantisation in making LLM serving practical.
- How to **detect** data drift and prediction drift in production environments.
- How to **interpret** key monitoring metrics and apply cost optimisation strategies for cloud AI deployments.

---

## Why Batch and Real-Time Serving Are Not Interchangeable

When you train a model in a notebook, you call `predict()` once and get an answer. In production, every millisecond counts, and the way you serve predictions fundamentally shapes your system architecture. **Batch inference** collects requests over a window of time and processes them together, maximising throughput and hardware utilisation. This is ideal for use cases like nightly recommendation refreshes, monthly credit risk scoring, or periodic anomaly reports where results are not needed instantly. **Real-time inference** via REST APIs, on the other hand, processes each request as it arrives and returns a response synchronously — essential for chatbots, fraud detection, and any application where users expect immediate feedback.

The decision between batch and real-time is not purely technical; it has direct cost and complexity implications. Batch pipelines can use cheaper CPU instances and scale horizontally with simple job orchestrators like Airflow or AWS Batch. Real-time systems require GPU-backed endpoints, load balancers, and auto-scaling policies that keep latency predictable under traffic spikes. Some architectures combine both: a real-time endpoint for high-priority requests and a batch system for bulk processing, with a routing layer that decides where each request goes. Understanding this tradeoff is the first decision you will make when designing any production ML system.

## How vLLM and Quantisation Make LLM Serving Feasible

Large language models pose a serving challenge that traditional ML model serving tools were never designed for. A 7-billion-parameter model in FP16 requires 14 GB of GPU memory just for the weights, and the **key-value cache** — which stores attention computations for each token in the generated sequence — can consume even more as context grows. **vLLM** solves this with PagedAttention, a memory management technique that splits the KV cache into fixed-size blocks, eliminating fragmentation and allowing memory to be shared across concurrent requests. This means a single GPU can serve multiple LLM requests simultaneously, dramatically improving throughput and reducing cost per token.

**Quantisation** takes a different approach by shrinking the model itself. Instead of storing each weight as a 32-bit or 16-bit float, **INT8** and **INT4 quantisation** map weights to 8-bit or 4-bit integers, reducing memory footprint by 2x to 4x with minimal accuracy degradation. A 7B model that needs 14 GB in FP16 can fit in under 4 GB with INT4 quantisation, opening the door to serving on cost-effective hardware. The tradeoff is not zero — quantisation introduces noise that can affect rare or edge-case predictions — but for most production applications, the efficiency gains far outweigh the precision loss. Tools like vLLM, TensorRT-LLM, and llama.cpp make these optimisations accessible without requiring deep systems programming.

## Where Production AI Serving and Monitoring Appears in Real Life

A fintech company processing thousands of loan applications per hour cannot afford a model that returns predictions in seconds — every millisecond of latency affects customer drop-off rates and regulatory compliance windows. They deploy quantised gradient-boosted models behind auto-scaled REST endpoints with batch fallback for non-urgent applications, and monitor prediction drift closely because shifts in applicant demographics or economic conditions can silently invalidate their risk models. In healthcare, radiology AI models must be served with sub-second latency during diagnostic workflows and monitored for data drift when a hospital installs a new CT scanner whose image characteristics differ from the training data — a change that could reduce diagnostic accuracy without any obvious error signal.

E-commerce platforms serving personalised product recommendations blend batch and real-time patterns: a nightly batch pipeline generates candidate recommendations for each user, while a real-time reranking model adjusts the final list based on the user's current session behaviour. Their monitoring dashboards track recommendation click-through rates as a proxy for prediction quality, flagging drops that might indicate drift. SaaS companies offering LLM-powered features — summarisation, code generation, customer support — rely heavily on vLLM-based serving with quantised models to keep infrastructure costs manageable while serving millions of daily requests. Across every industry, the common thread is clear: the ability to serve models reliably, detect when they degrade, and optimise infrastructure costs is what separates a proof-of-concept from a product that generates real business value.

## What's Next

After this session, you will be able to:
- Choose between batch and real-time serving patterns based on latency, throughput, and cost requirements.
- Deploy large language models using vLLM and apply INT8 or INT4 quantisation to reduce memory footprint.
- Set up data drift and prediction drift monitors to detect distribution shifts before they impact users.
- Design monitoring dashboards that track latency, throughput, error rates, and GPU utilisation.
- Apply cloud cost optimisation strategies including right-sizing, spot instances, and auto-scaling policies.
- Evaluate the accuracy-efficiency tradeoffs when quantising models for production deployment.

You do not need to have a production deployment running right now. The goal is to think like an ML engineer who deploys and maintains models in the real world, not just trains them in notebooks.

## Interesting Questions for the Live Session

- If quantising an LLM from FP16 to INT4 saves 75% on GPU costs but introduces a 1% accuracy drop, how do you decide whether the tradeoff is acceptable for your specific use case?
- Data drift detection sounds definitive in theory, but in practice how do you distinguish between a genuine distribution shift requiring retraining and a temporary seasonal pattern?
- Would you rather deploy a single large model that handles all request types or multiple smaller specialised models, and what operational complexity does each approach introduce?
- If your monitoring dashboard shows rising latency and falling accuracy simultaneously, how do you determine whether the root cause is data drift, hardware degradation, or a traffic pattern change?

By the end of this session, production AI serving should feel less like an afterthought and more like the core engineering discipline that transforms trained models into real impact: **Deployment is not the end of the pipeline — it is where the pipeline begins.**
