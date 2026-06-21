import tiktoken
from GPT.data.tokenizer import tokenise
from GPT.utils.logging import get_logger

logger=get_logger(__name__)

def build_samples(text,block_size):
    tokens=tokenise(text)
    samples=[]
    stride=block_size

    for i in range(0,len(tokens)-block_size,stride):
        chunk=tokens[i:i+block_size]
        samples.append(chunk)
    
    logger.info("Total samples:", len(samples))
    logger.info("Train samples:", int(0.8 * len(samples)))
    logger.info("Val samples:", len(samples) - int(0.8 * len(samples)))

    return samples