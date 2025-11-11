# SUMMARY

## Problem Statement

The problem discussed in this paper is bottlenecking of memory due to inefficient allocation of KV cache.

## Importance

Memory bottlenecking is a very serious problem for LLMs as most of them need to serve a lot of requests at a single time, but high memory usage of the KV cache limits this number even if you have a very powerful GPU and huge memory.
Since there is an entire industry around LLMs like GPT and PaLM, there is a strong incentive to serve as many requests as possible at once in the least amount of cost to maximize profits. Recent estimates show that the cost for an LLM request is about 10x more than a normal keyword query.

## Main Insight

At the core of an LLM lies an autoregressive transformer which generates one word at a time, and to generate a word it needs to look at the entire sentence every time. To improve that, we use memory caching of key and value matrices since after every token generation, there is just an addition of a new row and column and the rest of the matrix remains the same (called KV caching). Bottlenecks are caused by heavy memory pre-allocation to the KV cache. To improve this system, we use our understanding of operating systems. Hence, the solution proposed in this paper draws inspiration from concepts such as paging and virtual memory, creating a new attention algorithm called PagedAttention.

## Brief Overview

The solution to the problem is mainly that instead of pre-allocating the entire contiguous block for KV cache, memory is divided into smaller blocks which are dynamically allocated as per the number of tokens required by the LLMs.

# TECHNICAL UNDERSTANDING

## Problem Analysis

Existing models struggle because the KV cache memory, when managed inefficiently, grows linearly as the request increases. This memory is wasted in fragmentation and redundant duplication. Since this memory is linearly dependent on the batch size, it limits the batch size of the llm. Earlier models needed the entire KV cache to be stored in contiguous space in memory. This was the reason why they had to allocate memory required to store the maximum number of tokens to each request, which was mostly never used.

## Bottlenecks

The main causes of memory bottleneck in earlier models were internal fragmentation and external fragmentation.

- **Internal Fragmentation:** Since the number of tokens in each request are unknown and the memory needed to be allocated to each KV cache had to be contiguous, models allocated the maximum allowed tokens for each request. The actual required number of tokens used is usually much lower, resulting in a lot of wasted memory, as the unused memory cannot be allocated to a different request and remains blocked until the specific request clears up.
- **External Fragmentation:** Since pre-allocated memory for each request is different, it causes external fragmentation as well.
- **Reserved Memory:** Reserved memory contains keywords that are about to be outputted by the system (e.g., the eos which ends the iteration of an llm). Even though it is not a complete loss of memory, reserving this space for the entirety of the request is inefficient, as that block could be used for another request.

## Motivating Examples

The proposed solution to the problem of inefficient memory management is motivated by the classic idea of paging. Paging is a technique which divides virtual and physical memory into blocks to enable flexible memory allocation.

## Proposed Solution

The solution proposed in the paper uses a technique called PagedAttention, inspired by the classical techniques of paging and virtual memory in operating systems. This allows storing continuous pieces of information in non-contiguous memory, reducing the memory constraints of the KV cache by dividing the entire cache size into blocks which are dynamically allocated instead of blocking the entire memory at the time of request generation. This enables parallel computation of multiple requests on the GPU. Each KV block stores keys and values; to manage the KV cache, we use a KV cache manager which physically allocates the memory in the GPU, allowing the LLM to use the memory as if it were contiguous even if it isn't.

## Working

A request's KV cache is represented using various KV blocks like pages in an OS. When a request sends some input, it gets divided so it can be stored in blocks; the location of blocks in physical memory can be random but there is always a pointer in the GPU which stores the allocation of a block of memory in the KV cache. The last KV block's empty spaces are reserved for future keys and values. Virtually, the KV blocks are contiguous but physically they are not.
The KV block manager manages a block table which stores the pointers that connect the physical KV block to its virtual counterpart.
The most important feature of PagedAttention is that KV blocks are dynamically allocated.
Instead of working individually, a vLLM is deployed on top of other LLMs and it uses its own kernel to optimize the performance of the existing kernels.

## Implementations

- **Parallel Sampling:** When LLM assistants generate multiple sampled outputs for a single input, we replicate the last allocated KV block of inputs and generate two virtual KV caches independent of each other. Block tables generated for both are common until the point of separation, after which they have separate block tables. This method is similar to the copy-on-write technique in OSes (forking a process).
- **Beam Search:** In LLM tasks like machine translation, the k most appropriate results are taken. Implementation uses:
    1. Forking - creates a new sequence from the existing one and computes it in parallel
    2. Freeing - frees the KV cache of the sequence that falls off the beam

## Scheduling

The scheduling method followed is first-come-first-serve for all requests.
vLLMs follow the all-or-nothing policy in case the GPU runs out of memory for newer requests, i.e. it either keeps the entire request or evicts the entire request.
There are two main tactics used to recover evicted blocks by vLLMs:

1. **Swapping** - During eviction, the LLM copies the KV cache of the evicted request to CPU memory, freeing memory in the GPU for new requests (good for large requests).
2. **Recomputation** - When an evicted request is needed again, it just generates from scratch (good for small requests).
A single centralized scheduler manages memory of all the GPU workers; in this process, each GPU works independently and does a small part of a matrix multiplication which is constantly synchronized using all-reduce operations.

## Evaluation

- **Experimental Setup:**
The evaluation uses multiple OPT models (13B, 66B, 175B parameters) and LLaMA-13B on NVIDIA A100 GPUs via Google Cloud Platform instances. The 13B and 66B models represent popular production sizes, while 175B matches GPT-3's scale. Specific configurations include 1 A100 (40GB) for 13B models, 4 A100s (160GB) for 66B models, and 8 A100-80GB GPUs (640GB) for 175B models.

It uses ShareGPT and Alpaca Datasets.

- **Baselines:**
  - FasterTransformer: A highly optimized inference engine. vLLM team implemented a custom FCFS scheduler with dynamic batching and maximum batch sizes limited by GPU memory.
  - Orca Max: Reserves space for maximum sequence length (2048 tokens), representing naive memory management.
  - Orca Pow2: Reserves output space with 2× rounding (e.g. output of 25 tokens reserves 32 positions).
  - Orca Oracle: Assumes perfect knowledge of output lengths, representing an upper-bound baseline.

- **Key Results and Metrics:**
The evaluation focuses on normalized latency (end-to-end latency divided by output length in tokens/second). vLLM is evaluated at different request rates until the system saturates, with measurements over 1-hour traces (15 minutes for 175B due to cost constraints).

1. Basic Sampling (Single Output per Request)
vLLM achieves **2-4× throughput improvements** over state-of-the-art systems with comparable latency.

- ShareGPT dataset: vLLM sustains **1.7-2.7× higher request rates** compared to Orca Oracle and **2.7-8× higher** than Orca Max
- Batch size improvements: For OPT-13B, vLLM processes **2.2× more requests** simultaneously than Orca Oracle and **4.3× more** than Orca Max
- Vs FasterTransformer: Up to **22× higher request rates**

The advantages are pronounced with long sequences due to higher memory waste in existing systems.

- **Memory Fragmentation Breakdown:**
  - vLLM: Only **13.6% memory waste**, including token states and reserved/fragmented space
  - Orca Oracle: 20.4% waste (best baseline)
  - Orca Pow2: 57.3% waste
  - Orca Max: 96.3% waste

This demonstrates that inefficient memory management artificially limits batch sizes in competing systems.

- **Complex Decoding Algorithms:**
  - Parallel Sampling: 6.1-9.8% memory savings on Alpaca, **16.2-30.5%** on ShareGPT
  - Beam Search: 37.6-55.2% memory savings on Alpaca, **44.3-66.3%** on ShareGPT; vLLM throughput improvement over Orca Oracle: **1.3× for basic sampling, increasing to 2.3×** for beam width 6
  - Shared Prefix Scenario: For machine translation tasks with few-shot examples, **1-shot prefix:** vLLM achieves **1.67× higher throughput**, **5-shot prefix:** **3.58× higher throughput**
  - Chatbot Workload: On ShareGPT data, vLLM sustains **2× higher request rates** compared to all three Orca baselines

## How Results Support Claims

1. **Memory efficiency claim:** The 13.6% waste vs. 20.4-96.3% in baselines directly validates PagedAttention's efficient memory utilization.
2. **Throughput claim (2-4×):** Consistent improvements across models, datasets, and decoding algorithms prove generality of the approach.
3. **Complex decoding support:** Savings scale with sharing opportunity, validating copy-on-write and reference counting.
4. **Fairness in scheduling:** FCFS policy with all-or-nothing preemption ensures earliest requests complete first.
5. **Distributed scalability:** Scaling from 13B (1 GPU) to 175B (8 GPUs) with consistent improvements demonstrates generalization to distributed settings.

# CRITICAL ANALYSIS

## Strengths

- The paper demonstrates exceptional ingenuity by connecting OS concepts and LLM performance. The authors successfully apply **paging, virtual-to-physical address mapping, copy-on-write, and reference counting** into a novel GPU attention algorithm (PagedAttention), solving a modern, domain-specific problem with remarkable effectiveness.
- The experiments were rigorously validated, comparing memory occupancy with existing systems using real datasets and multiple competitive baselines.
- Diverse workload scenarios (sampling, beam search, parallel sampling, shared prefixes, chatbot applications) were evaluated across multiple models and scales.
- Rigorous ablations examined kernel overhead (20-26% for PagedAttention vs. 2-4× end-to-end speedup), block size selection, and recovery mechanisms (swapping vs. recomputation).

## Weaknesses

- **Kernel Overhead Not Fully Amortized:** PagedAttention incurs a 20-26% increase in attention kernel latency versus highly optimized baselines. Block table indirection, branch instructions, and variable length handling are persistent limitations.
- **Scheduling Policy Limitations:** FCFS scheduling with all-or-nothing preemption is fair but suboptimal for production deployments needing SLAs, priority support, and more nuanced fairness guarantees.
- **Single GPU Family Evaluation:** Only NVIDIA A100s tested; generalization to other hardware (H100, consumer GPUs) unexplored.
- **Short trace duration for largest model:** Only 15-minute traces for 175B model, reducing confidence in sustained performance.
- **No formal verification of correctness properties.**
- **Internal fragmentation:** Even with 16-token blocks, misaligned sequences waste the final block's slots.
- **No adaptive block sizing:** All blocks are fixed-size, missing potential savings for variable-length workloads.

## Specific Critique

### 1. Is the system practical to deploy?

The system is practical and is being used widely by LLMs (including OpenAI and Gemini). It is open source, maintainable (8.5K lines of Python and 2K lines of CUDA C), and integrates seamlessly with existing APIs. vLLM's 2-4× throughput gains translate to cost savings, making it financially attractive for deployment.

### 2. Are the claimed contributions novel?

Yes, the authors' claims are novel: while they leverage existing OS concepts, applying them to LLM KV cache management yields a groundbreaking result in this domain.

### 3. Does the solution generalize beyond the tested scenarios?

No, the solution mostly applies to transformer-based LLM inference. Its advantages do not translate to general GPU workloads with static tensors or compute-bound tasks.

# PERSONAL REFLECTION

## What I Learned

The most profound learning from this paper is how **elegant system design bridges theory and practice**. The core contribution—applying OS virtual memory concepts to GPU memory management—sounds simple but is executed with deep systems thinking. vLLM doesn't just borrow paging; it adapts it thoughtfully with block-granular copy-on-write, all-or-nothing preemption, and LLM-specific recovery mechanisms.

I also learned that **addressing memory fragmentation is fundamentally more impactful than optimizing computation**. Existing systems waste 60-96% of GPU memory due to fragmentation, yet kernel optimizations alone cannot fix this. vLLM's 2-4× throughput gain is primarily due to reducing wasted memory, not kernel speed.

## What Surprised Me

**The elegance of recomputation as a memory recovery mechanism** was surprising. Intuitively, swapping (storing to CPU RAM) seems superior—you're "saving" computed state. However, recomputation is often better because regenerating KV cache during the highly parallelizable prefill phase is cheaper than retrieving from CPU RAM, which incurs PCIe bandwidth costs. Sometimes, **it's cheaper to recompute than to store**.

## How does this relate to other systems concepts you know?

vLLM is a clear example of translating virtual memory and paging concepts from OS into an LLM serving context. Block tables resemble page tables, copy-on-write echoes process forking in Unix, and resource scheduling repurposes classical FCFS. The integration of kernel, scheduler, and algorithm reflects modular co-design found in database and distributed operating systems.

## Would you have designed it differently? How?

- **Adaptive Block Sizing:** Rather than fixed block size 16, dynamically tune block sizes depending on request properties and historic token length distribution.
- **Priority-Aware Scheduling:** Incorporate lightweight priorities based on SLA requirements, estimated output length, and arrival time.
- **Learned Recovery Strategy:** Use ML models to select between swapping and recomputation.
- **Multi-Tenancy Isolation:** Per-tenant quotas and shared cache for common prefixes, balancing throughput and QoS.
- **Hardware/Software Co-Design:** Advocate for enhanced GPU primitives supporting efficient paging and block-level addressing.

**Overall, vLLM and PagedAttention are excellent examples of creative systems thinking, combining OS concepts, memory management, GPU programming, and deep learning efficiency.**

