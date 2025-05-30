import tiktoken

# Initialize the encoder (using cl100k_base which is used by GPT-4)
encoder = tiktoken.get_encoding("cl100k_base")

# Count tokens
text = """Fish and Game remarked:\n\n> \u201cWe buy and process feathered and ground game from hunters throughout Cornwall and Devon. Game meat is a product area which is becoming more and more popular and is one of the few product ranges, along with fish, that can be guaranteed wild and traceable. Another factor responsible for increasing the"
"""

# Encode the text and count tokens
tokens = encoder.encode(text)
print(f"Number of tokens: {len(tokens)}")