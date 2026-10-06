# GPT Model Flowchart

This diagram summarizes the TensorFlow GPT decoder implemented in `GPT/model/gpt.py`
and its supporting layers under `GPT/layers/`.

```mermaid
flowchart TD
    A["Input token ids<br/>(batch, sequence)"] --> B["Token embedding<br/>WTE: vocab_size -> n_embedd"]
    A --> C["Position ids<br/>0..sequence-1"]
    C --> D["Position embedding<br/>WPE: block_size -> n_embedd"]
    B --> E["Add token + position embeddings<br/>(batch, sequence, n_embedd)"]
    D --> E

    E --> F["DecoderBlock x n_layer<br/>pre-norm transformer stack"]

    subgraph BLOCK["DecoderBlock"]
        F1["LayerNorm ln_1"] --> F2["Masked multi-head self-attention"]
        F2 --> F3["Residual add"]
        F3 --> F4["LayerNorm ln_2"]
        F4 --> F5["Feed-forward network<br/>Dense d_ff -> GELU -> Dense n_embedd"]
        F5 --> F6["Residual add"]
    end

    F --> G["Final LayerNorm ln_f"]
    G --> H["Output projection<br/>matmul with WTE embeddings transpose"]
    H --> I["Logits<br/>(batch, sequence, vocab_size)"]
```

## Attention Detail

```mermaid
flowchart TD
    A["Block input<br/>(B, T, C)"] --> B["Combined QKV projection<br/>Dense 3C"]
    B --> C["Split Q, K, V"]
    C --> D["Reshape and transpose<br/>(B, heads, T, head_dim)"]
    D --> E["Attention scores<br/>QK^T / sqrt(head_dim)"]
    E --> F["Add causal mask"]
    F --> G["Softmax"]
    G --> H["Weighted sum with V"]
    H --> I["Merge heads<br/>(B, T, C)"]
    I --> J["Output projection<br/>Dense C"]
```

## Default Configuration Shape

| Component | Default |
| --- | --- |
| Vocabulary size | 50257 |
| Block size | 128 |
| Layers | 12 |
| Attention heads | 12 |
| Embedding width | 768 |
| Feed-forward width | 3072 |
