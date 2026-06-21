import tiktoken
from GPT.data.tokenizer import tokenise

def build_samples(text,block_size):
    tokens=tokenise(text)
    samples=[]
    stride=block_size

    for i in range(0,len(tokens)-block_size,stride):
        chunk=tokens[i:i+block_size]
        samples.append(chunk)
    
    print("Total samples:", len(samples))
    print("Train samples:", int(0.8 * len(samples)))
    print("Val samples:", len(samples) - int(0.8 * len(samples)))

    return samples