import string

def encode(data: string) -> bytes:
    """Encode a string to bytes using UTF-8 encoding."""
    return data.encode('utf-8') 

encode_result = encode("Hello, World!")
print(encode_result)  # Output: b'Hello, World!'