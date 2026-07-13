#GPT/inference/sampler.py

import tensorflow as tf
import numpy as np
from GPT.config import GPTConfig
from typing import List, Tuple, Optional


SUPPORTED_DECODE_STRATEGIES = {"greedy", "sample", "top_k", "top_p"}

class Sampler:
    '''Text generation sampler for GPT model.'''

    def __init__(self, model=None, config: Optional[GPTConfig] = None):
        '''
        Initialize the sampler with optional model and configuration.

        Args:
            model: GPT model used for generation. Optional for convenience wrappers.
            config: Sampling configuration. Defaults to GPTConfig() when omitted.
        '''
        self.model = model
        self.config = config or GPTConfig()
        self.default_temperature = self.config.temperature
        self.default_top_k = self.config.top_k
        self.default_top_p = self.config.top_p

    @staticmethod
    def temperature_scaling(logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        '''
        Apply temperature scaling to logits.
        Args:
            logits: Tensor of shape (batch_size, vocab_size)
            temperature: Temperature value for scaling

        Temperature < 1.0: Makes model more confident (sharper distribution)
        Temperature = 1.0: No change (neutral)
        Temperature > 1.0: Makes model less confident (softer distribution)

        Returns:
            Scaled logits

        Raises:
            ValueError: If temperature <= 0
        '''
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")
        return logits / temperature

    @staticmethod
    def top_k_filtering(logits: tf.Tensor, k: int = 50) -> tf.Tensor:
        """Apply top-k filtering: keep only top-k highest probability tokens.

        Sets all non-top-k logits to very negative value so softmax ignores them.

        Args:
            logits: (vocab_size,) logits
            k: Number of top tokens to keep

        Returns:
            Filtered logits with non-top-k set to -1e10

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        vocab_size = tf.shape(logits)[0]
        k = tf.minimum(k, vocab_size)
        
        # Get top-k values and indices
        top_k_logits, top_k_indices = tf.nn.top_k(logits, k=k, sorted=False)
        
        # Create mask for top-k indices
        mask = tf.scatter_nd(
            tf.expand_dims(top_k_indices, 1),
            tf.ones(k, dtype=tf.bool),
            [vocab_size]
        )
        
        # Set non-top-k to very negative value
        filtered_logits = tf.where(
            mask,
            logits,
            tf.fill(tf.shape(logits), -1e10)
        )
        
        return filtered_logits
    
    @staticmethod
    def top_p_filtering(logits: tf.Tensor, p: float = 0.95) -> tf.Tensor:
        """Apply top-p (nucleus) sampling: keep tokens with cumulative prob >= p.

        Keeps the smallest set of tokens whose cumulative probability exceeds p.

        Args:
            logits: (vocab_size,) logits
            p: Cumulative probability threshold (0 < p <= 1)

        Returns:
            Filtered logits with non-nucleus tokens set to -1e10

        Raises:
            ValueError: If p not in (0, 1]
        """
        if not (0 < p <= 1):
            raise ValueError(f"p must be in (0, 1], got {p}")
        
        vocab_size = tf.shape(logits)[0]
        
        # Sort in descending order
        sorted_logits, sorted_indices = tf.nn.top_k(logits, k=vocab_size, sorted=True)
        
        # Compute cumulative probabilities
        sorted_probs = tf.nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = tf.cumsum(sorted_probs, axis=-1)
        
        # Find indices where cumsum <= p
        indices_to_remove = cumsum_probs <= p
        
        # Keep at least one token (the first/most probable)
        indices_to_remove = tf.concat([
            [False],
            indices_to_remove[:-1]
        ], axis=0)
        
        # Create mask from sorted indices
        sorted_logits_filtered = tf.where(
            indices_to_remove,
            sorted_logits,
            tf.fill(tf.shape(sorted_logits), -1e10)
        )
        
        # Unsort back to original order
        unsort_indices = tf.argsort(sorted_indices)
        filtered_logits = tf.gather(sorted_logits_filtered, unsort_indices)
        
        return filtered_logits
    
    def _decode_next_token(
        self,
        model,
        generated: List[int],
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_top_k: bool = True,
        use_top_p: bool = False,
        decode_strategy: str = "sample",
        block_size: int = 1024,
    ) -> int:
        """Decode one token from the model using a shared strategy switch."""
        temperature = self.default_temperature if temperature is None else temperature
        top_k = self.default_top_k if top_k is None else top_k
        top_p = self.default_top_p if top_p is None else top_p

        if decode_strategy not in SUPPORTED_DECODE_STRATEGIES:
            raise ValueError(
                f"Unsupported decode_strategy '{decode_strategy}'. "
                f"Expected one of {sorted(SUPPORTED_DECODE_STRATEGIES)}"
            )

        context = generated[-block_size:]
        context_tensor = tf.constant([context], dtype=tf.int32)

        logits = model(context_tensor, training=False)
        next_logits = logits[0, -1, :]
        next_logits = self.temperature_scaling(next_logits, temperature)

        if decode_strategy == "greedy":
            return int(tf.argmax(next_logits).numpy())

        if use_top_p:
            next_logits = self.top_p_filtering(next_logits, p=top_p)
        elif use_top_k or decode_strategy in {"top_k", "top_p"}:
            next_logits = self.top_k_filtering(next_logits, k=top_k)

        if decode_strategy == "top_p":
            next_logits = self.top_p_filtering(next_logits, p=top_p)

        probs = tf.nn.softmax(next_logits)
        if decode_strategy in {"sample", "top_k", "top_p"}:
            next_token = tf.random.categorical(
                tf.math.log(probs[tf.newaxis, :]),
                num_samples=1,
            )[0, 0].numpy()
            return int(next_token)

        raise ValueError(f"Unsupported decode_strategy '{decode_strategy}'")

    def generate(
        self,
        model,
        input_ids: tf.Tensor,
        max_new_tokens: int = 50,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_top_k: bool = True,
        use_top_p: bool = False,
        block_size: int = 1024,
        decode_strategy: str = "sample",
    ) -> tf.Tensor:
        """Generate text using a shared decode loop with configurable strategy.

        Args:
            model: GPT-2 model that outputs logits.
            input_ids: (1, seq_len) starting token IDs.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Keep only top-k highest probability tokens.
            top_p: Keep tokens with cumulative prob >= top_p.
            use_top_k: Whether to use top-k filtering.
            use_top_p: Whether to use top-p filtering (overrides top-k if True).
            block_size: Maximum sequence length for model.
            decode_strategy: One of {'greedy', 'sample', 'top_k', 'top_p'}.
        """
        if hasattr(input_ids, "numpy"):
            generated = input_ids.numpy().tolist()[0]
        else:
            generated = list(input_ids[0])

        for _ in range(max_new_tokens):
            next_token = self._decode_next_token(
                model,
                generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_top_k=use_top_k,
                use_top_p=use_top_p,
                decode_strategy=decode_strategy,
                block_size=block_size,
            )
            generated.append(next_token)

        return tf.constant([generated], dtype=tf.int32)

    def greedy_decode(
        self,
        model,
        input_ids: tf.Tensor,
        max_new_tokens: int = 50,
        block_size: int = 1024,
    ) -> tf.Tensor:
        """Greedy decoding: always pick the most likely next token."""
        return self.generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            decode_strategy="greedy",
        )
    
    def beam_search(
        self,
        model,
        input_ids: tf.Tensor,
        max_new_tokens: int = 50,
        beam_width: int = 4,
        block_size: int = 1024,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Beam search: keep top-k hypotheses and extend each.
        
        More expensive but higher quality output.
        
        Args:
            model: GPT-2 model
            input_ids: Starting tokens
            max_new_tokens: Maximum tokens to generate
            beam_width: Number of beams to keep
            block_size: Maximum sequence length
        
        Returns:
            Tuple of (best sequence, log probability)
        """
        sequences = [(input_ids.numpy().tolist()[0], 0.0)]
        
        for step in range(max_new_tokens):
            all_candidates = []
            
            for seq, score in sequences:
                context = seq[-block_size:]
                context_tensor = tf.constant([context], dtype=tf.int32)
                
                logits = model(context_tensor, training=False)
                log_probs = tf.nn.log_softmax(logits[0, -1, :])
                
                # Get top beam_width tokens
                top_log_probs, top_indices = tf.nn.top_k(log_probs, k=beam_width)
                
                for log_prob, idx in zip(top_log_probs.numpy(), top_indices.numpy()):
                    new_seq = seq + [int(idx)]
                    new_score = score + float(log_prob)
                    all_candidates.append((new_seq, new_score))
            
            # Keep top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = all_candidates[:beam_width]
        
        best_seq, best_score = sequences[0]
        return tf.constant([best_seq], dtype=tf.int32), tf.constant(best_score)
 
 
 
def generate_with_temperature(
    model,
    input_ids: tf.Tensor,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
) -> tf.Tensor:
    """Generate text with temperature sampling only."""
    sampler = Sampler()
    return sampler.generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        decode_strategy="sample",
    )


def generate_with_top_k(
    model,
    input_ids: tf.Tensor,
    temperature: float = 0.7,
    top_k: int = 40,
    max_new_tokens: int = 50,
) -> tf.Tensor:
    """Generate text with temperature + top-k sampling."""
    sampler = Sampler()
    return sampler.generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_top_k=True,
        decode_strategy="top_k",
    )


def generate_with_top_p(
    model,
    input_ids: tf.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 50,
) -> tf.Tensor:
    """Generate text with temperature + top-p (nucleus) sampling."""
    sampler = Sampler()
    return sampler.generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_top_p=True,
        decode_strategy="top_p",
    )


def greedy_generate(
    model,
    input_ids: tf.Tensor,
    max_new_tokens: int = 50,
) -> tf.Tensor:
    """Generate text greedily (deterministic)."""
    sampler = Sampler()
    return sampler.greedy_decode(model, input_ids, max_new_tokens=max_new_tokens)
 