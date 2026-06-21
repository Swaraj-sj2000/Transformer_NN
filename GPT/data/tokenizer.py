import tiktoken
from GPT.utils.logging import get_logger
logger=get_logger(__name__)

def tokenise(text,model="gpt2"):
    logger.info("Tokenizing...") 
    enc=tiktoken.get_encoding(model)
    tokens=enc.encode(text)
    logger.info("Finished tokenizing....")
    return tokens

    