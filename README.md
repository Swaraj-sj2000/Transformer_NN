# GPT From Scratch in TensorFlow

This repository contains a compact GPT-2 style decoder-only transformer implemented with TensorFlow/Keras. It is organized as a learning-oriented project: each part of the model, data pipeline, training loop, and sampling stack lives in its own module and has focused tests around the important tensor shapes and behaviors.

The project currently targets small-scale experimentation, especially with the Tiny Shakespeare dataset. It includes causal multi-head self-attention, token and positional embeddings, decoder blocks, tied output projection, TFRecord dataset generation, training utilities, checkpoint helpers, and text generation sampling strategies.

## What Is Included

- Decoder-only GPT model implemented in `GPT/model/gpt.py`
- Token and positional embeddings in `GPT/layers/embedding.py`
- Causal multi-head self-attention in `GPT/layers/attention.py`
- Pre-norm transformer decoder blocks in `GPT/layers/block.py`
- Feed-forward network implementations in `GPT/layers/mlp.py`
- Config validation with Pydantic settings in `GPT/config.py`
- GPT-2 BPE tokenization with `tiktoken`
- Tiny Shakespeare download and TFRecord sharding pipeline
- Sparse categorical cross-entropy next-token loss
- Cosine learning-rate schedule with warmup
- Optimizer builder for Adam, SGD, AdaGrad, RMSProp, Lion, and Adafactor
- Inference sampler with greedy, random, top-k, and top-p decoding paths
- Unit tests for model layers, loss, sampler behavior, records, and model flow

## Repository Layout

```text
GPT/
  config.py                     Project configuration and validation
  configs/                      Placeholder YAML config files
  data/                         Dataset download, tokenization, sample building, TFRecords
  docs/                         Additional design and progress notes
  exceptions/                   Custom exception classes
  inference/                    Sampling and generation utilities
  layers/                       Embedding, attention, MLP, and decoder block layers
  model/                        Full GPT decoder model
  notebooks/                    Exploratory notebooks and notebook refactor notes
  tests/                        Pytest test suite
  training/                     Loss, optimizer, scheduler, trainer, evaluation loop
  utils/                        Logging, checkpoint, seed, and debugging helpers
```

## Architecture Overview

The main model class is `Decoder` in `GPT/model/gpt.py`. It follows the standard GPT decoder flow:

```text
input token ids
  -> token embeddings + positional embeddings
  -> repeated decoder blocks
  -> final layer normalization
  -> tied projection through token embedding weights
  -> vocabulary logits
```

Each decoder block uses pre-normalization:

```text
x -> LayerNorm -> causal self-attention -> residual add
  -> LayerNorm -> feed-forward network -> residual add
```

The attention implementation projects the input into Q, K, and V with one combined dense layer, reshapes into attention heads, applies a causal lower-triangular mask, computes softmax attention, merges heads, and applies the output projection.

The output head is weight-tied with the token embedding matrix:

```python
logits = tf.matmul(x, self.embedding.wte.embeddings, transpose_b=True)
```

## Configuration

Runtime settings live in `GPT/config.py` as a `GPTConfig` Pydantic settings model. Values can be provided directly in Python or through environment variables using the `GPT_` prefix.

Important defaults:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `vocab_size` | `50257` | GPT-2 tokenizer vocabulary size |
| `block_size` | `128` | Maximum sequence length |
| `n_layer` | `12` | Number of decoder blocks |
| `n_head` | `12` | Number of attention heads |
| `n_embedd` | `768` | Embedding width |
| `d_ff` | `3072` | Feed-forward hidden width |
| `batch_size` | `8` | Training batch size |
| `learning_rate` | `3e-4` | Base learning rate |
| `warmup_steps` | `2000` | Warmup steps for scheduler |
| `total_steps` | `100000` | Total schedule length |
| `temperature` | `1.0` | Sampling temperature |
| `top_k` | `50` | Top-k sampling cutoff |
| `top_p` | `0.9` | Nucleus sampling threshold |

Validation is built in for core constraints such as:

- `block_size` must be a power of two.
- `n_embedd` must be divisible by `n_head`.
- `warmup_steps` must be smaller than `total_steps`.
- Data, checkpoint, and log directories are created automatically when the config is initialized.

Example override:

```bash
GPT_BLOCK_SIZE=256 GPT_BATCH_SIZE=4 python -m GPT.config
```

## Setup

This project does not currently include a pinned dependency file. A practical development environment needs Python 3.10+ and the libraries imported by the codebase.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install tensorflow numpy pydantic pydantic-settings tiktoken requests tqdm pytest
```

Optional notebook and documentation helpers may require additional packages depending on which files you run, such as `jupyter`, `transformers`, or `reportlab`.

## Build The Dataset

The dataset pipeline downloads Tiny Shakespeare, tokenizes it with GPT-2 BPE, chunks it into fixed-length samples, and writes sharded TFRecords.

Run:

```bash
python -m GPT.data.build_dataset_pipeline
```

By default this creates:

```text
GPT/data/shakespeare.txt
GPT/data/tfrecords/train/*.tfrecord
GPT/data/tfrecords/val/*.tfrecord
```

The dataset reader in `GPT/data/tfrecord_reader.py` parses each serialized example into an `input_ids` tensor of shape:

```text
(block_size,)
```

Training batches therefore have shape:

```text
(batch_size, block_size)
```

## Train

After generating TFRecords, the training entry point is:

```bash
python -m GPT.training.trainer
```

The training step performs next-token prediction:

```text
inputs  = x[:, :-1]
targets = x[:, 1:]
```

The model receives the full batch, logits are shifted to match the targets, and `GPTLoss` computes sparse categorical cross-entropy from logits.

The optimizer is built through:

```python
from GPT.training.optimizer import build_optimizer
```

The learning rate uses `CosineWarmupSchedule`, which linearly warms up to the base learning rate and then decays with a cosine schedule.

Current note: `GPT/training/trainer.py` references `TrainLogger()` without importing or defining it, and imports `get_dataset` from `GPT.data.build_dataset_pipeline` even though the dataset reader defines it in `GPT.data.tfrecord_reader`. Fixing those two references is needed before the script can run end-to-end as a command-line trainer.

## Generate

The sampler supports:

- `greedy`
- `sample`
- `top_k`
- `top_p`

Basic example:

```python
import tensorflow as tf

from GPT.config import GPTConfig
from GPT.model.gpt import Decoder
from GPT.inference.sampler import Sampler

config = GPTConfig()
model = Decoder(config=config)

prompt = tf.constant([[1, 2, 3]], dtype=tf.int32)
_ = model(prompt, training=False)

sampler = Sampler(config=config)
generated = sampler.generate(
    model,
    prompt,
    max_new_tokens=8,
    decode_strategy="greedy",
    temperature=config.temperature,
    top_k=config.top_k,
    top_p=config.top_p,
)

print(generated.numpy().tolist())
```

There is also a small script that loads `checkpoints/model.weights.h5` when available:

```bash
python -m GPT.test_generation_after_load
```

If no checkpoint exists, it runs with randomly initialized weights and prints generated token ids.

## Tests

Run the test suite with:

```bash
pytest GPT/tests
```

Useful targeted runs:

```bash
pytest GPT/tests/test_sampler.py
pytest GPT/tests/test_attention.py
pytest GPT/tests/test_model.py
pytest GPT/tests/test_loss.py
```

Some tests instantiate TensorFlow models and may take longer on CPU-only machines.

## Development Notes

- The model enables TensorFlow mixed precision globally in `GPT/model/gpt.py` with `mixed_float16`.
- The implementation is intentionally modular, which makes it easy to inspect intermediate components in isolation.
- `GPT/configs/*.yaml` files are present as placeholders, but the active configuration system is the Pydantic `GPTConfig` class.
- `GPT/docs/model_flowchart.md` and `GPT/docs/module4_revision_and_next_steps.md` contain additional project notes.
- `GPT/notebooks/GPT_2_Clean.ipynb` contains notebook-based exploration.

## Known Status

The model, layer, loss, scheduler, optimizer, sampler, and dataset modules are present and organized for experimentation. The training command needs a small cleanup before it can be treated as a polished one-command training script:

- Import or replace the missing `TrainLogger`.
- Import `get_dataset` from `GPT.data.tfrecord_reader`.
- Consider adding a pinned `requirements.txt` or `pyproject.toml`.
- Consider wiring the placeholder YAML config files into the runtime configuration system or removing them.

## License

No license file is currently included. Add a license before distributing or reusing this project outside private experimentation.
