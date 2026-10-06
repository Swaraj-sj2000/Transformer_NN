# Module 4 Revision And Next Steps

Last attended tracker point:

Phase II - Training LLMs from Scratch
Module 4 - Pre-Training Pipeline

Status: completed, with revision recommended before moving to Module 5.

## What You Completed

| Tracker item | Repo evidence |
| --- | --- |
| GPT-2 small architecture in TF/Keras subclassing | `GPT/model/gpt.py`, `GPT/layers/` |
| Custom training loop with GradientTape | `GPT/training/trainer.py` |
| Loss, gradient clipping, training step | `GPT/training/loss.py`, `GPT/training/trainer.py` |
| Cosine LR schedule with warmup | `GPT/training/scheduler.py` |
| TFRecord token pipeline | `GPT/data/` |
| Mixed precision setup | `GPT/model/gpt.py`, notebook |
| Gradient checkpointing concept | `GPT/utils/checkpoint.py` |
| Perplexity evaluation loop | `GPT/training/evaluation_loop.py` |
| HuggingFace GPT-2 weight loading | `GPT/notebooks/GPT_2_Clean.ipynb` |
| Top-k, top-p, greedy generation | `GPT/inference/sampler.py` |
| Post-load generation smoke check | `GPT/test_generation_after_load.py` |
| Architecture diagram | `GPT/docs/model_flowchart.md` |

## Revision Pass

### 1. Forward Pass

Be able to explain this without reading code:

1. `input_ids` enter as shape `(batch, seq_len)`.
2. Token embeddings and position embeddings are added.
3. The hidden states pass through `n_layer` decoder blocks.
4. Each block uses pre-norm residual structure:
   - LayerNorm
   - causal multi-head self-attention
   - residual add
   - LayerNorm
   - feed-forward network
   - residual add
5. Final LayerNorm is applied.
6. Logits are produced with tied output projection using token embedding weights.

Quick check:

```text
Why does the model multiply by WTE transpose instead of using a separate output Dense layer?
```

Expected answer: it ties input token embeddings and output token classifier weights, reducing parameters and matching GPT-style language modeling.

### 2. Training Loop

Core mental model:

```text
batch tokens -> inputs/labels shift -> logits -> cross entropy -> gradients -> clip -> optimizer step
```

You should be able to answer:

- Why labels are shifted by one token.
- Why clipping helps when gradients spike.
- Why `@tf.function` speeds repeated train steps.
- What breaks if logits and labels have mismatched sequence lengths.

### 3. Learning Rate Schedule

Warmup avoids aggressive early updates before the model has stable gradient statistics.
Cosine decay lowers the learning rate smoothly after warmup.

Quick check:

```text
At step 0, LR should be near 0.
At warmup_steps, LR should be near base LR.
Near total_steps, LR should decay toward 0.
```

### 4. TFRecord Pipeline

Pipeline idea:

```text
raw text -> tokenizer -> fixed-length token windows -> TFRecord shards -> tf.data read/parse/shuffle/batch/prefetch
```

Why it matters:

- Shards make large datasets manageable.
- `tf.data` keeps the GPU fed.
- Fixed-length examples simplify batching and next-token labels.

### 5. Mixed Precision

Mixed precision speeds matmul-heavy models and reduces memory pressure.

Revision questions:

- Which tensors can safely use float16?
- Why do losses/softmax/logits sometimes need care for numerical stability?
- What is loss scaling, and why does it prevent underflow?

### 6. Gradient Checkpointing

Activation memory grows with:

```text
batch_size * sequence_length * hidden_size * number_of_layers
```

Checkpointing trades compute for memory:

```text
do not store some activations -> recompute them during backward pass
```

### 7. Sampling

Generation loop:

```text
prompt -> model logits -> choose next token -> append token -> repeat
```

Sampling modes:

- Greedy: choose argmax token.
- Temperature: sharpen or flatten logits.
- Top-k: sample only from the k strongest candidates.
- Top-p: sample from the smallest candidate set whose probability mass exceeds p.

## Loss Spike At Step 3000

Five likely causes:

1. Learning rate too high after warmup or schedule bug.
2. Gradient explosion, especially in attention or FFN projections.
3. Bad/corrupt batch from tokenization or TFRecord parsing.
4. Mixed precision overflow/underflow or missing loss scaling.
5. Checkpoint or pretrained-weight mismatch causing unstable resumed training.

How to debug:

1. Log LR, loss, gradient norm, and batch token stats at the spike.
2. Re-run the same step/batch with a lower LR.
3. Validate TFRecord parsing on the offending shard.
4. Temporarily disable mixed precision to compare behavior.
5. Verify checkpoint architecture and tensor shapes.

## Before Moving On

Do this short revision loop:

1. Read `GPT/docs/model_flowchart.md`.
2. Open `GPT/model/gpt.py` and trace one forward pass.
3. Open `GPT/training/trainer.py` and explain each line of `train_step`.
4. Open `GPT/inference/sampler.py` and explain greedy vs top-k vs top-p.
5. Run or mentally simulate the generation smoke script.

## Next Tracker Point

Phase II - Module 5: Distributed & Efficient Training

Tasks:

1. Data parallelism - `tf.distribute.MirroredStrategy`, gradient aggregation.
2. Model parallelism - tensor vs pipeline parallelism.
3. ZeRO optimizer stages - shard parameters, gradients, optimizer state.
4. Implement MirroredStrategy training on this GPT and verify gradient sync.
5. Tensor parallelism with `tf.distribute` - split attention heads across GPUs.
6. Gradient accumulation - simulate large batch on small GPU memory.
7. Communication bottlenecks - AllReduce, ring topology, bandwidth vs latency.
8. TFRecords sharding strategy for distributed data pipelines.
9. Check: with 8 A100s, how would you train a 70B model?

Recommended next implementation:

Start with gradient accumulation and MirroredStrategy because they extend the current trainer cleanly. Leave tensor parallelism and ZeRO as theory/design first, then implement smaller experiments once the single-GPU training loop is stable.
