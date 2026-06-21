import tiktoken

def tokenise(text,model="gpt2"):
    print("Tokenizing...") 
    enc=tiktoken.get_encoding(model)
    tokens=enc.encode(text)
    print("Finished tokenizing....")
    return tokens

    