import string

def encode(data: string) -> bytes:
    """Encode a string to bytes using UTF-8 encoding."""
    return data.encode('utf-8') 

def decode(data: bytes) -> string:
    """Decode bytes to a string using UTF-8 encoding."""
    return data.decode('utf-8')

encode_result = encode("Hello, World!")
print(encode_result)  # Output: b'Hello, World!'
decode_result = decode(encode_result)
print(decode_result)  # Output: Hello, World!

def encode_byte_mode(data: str) -> list[int]:
    """Encode a string to a list of byte values."""
    return list(data.encode('utf-8'))   
def decode_byte_mode(data: list[int]) -> str:
    """Decode a list of byte values to a string."""
    return bytes(data).decode('utf-8')
encode_byte_mode_result = encode_byte_mode("Hello, World!")
print(encode_byte_mode_result)  # Output: [72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33]