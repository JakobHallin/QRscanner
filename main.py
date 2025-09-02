import string


print("will try with qr")
def qr_encode_byte_mode(data: str) -> list[int]: #will only do 1L version
    """Encode a string to 1-L qr code byte mode."""
    bitstream = "0100"  # Mode indicator for byte mode //this tells we use byte
    
    bitstream += format(len(data), '08b')  # Character count indicator for my (8 bits for version 1 Byte mode)

    for char in data: #we need to convert each char to its byte value so i need to encode each char
        bitstream += format(ord(char), '08b')  # 8 bits per character we use ord to get the byte value of each char
    # Add terminator
    bitstream += "0000"  # Terminator (4 bits)
    # Pad to make the length a multiple of 8
    while len(bitstream) % 8 != 0:
        bitstream += "0" # Pad with 0s so we can get full bytes
    
    #  Convert to integers (codewords)
    codewords = [int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8)]

    # Pad with 0xEC and 0x11 to reach the required length for version 1-L (19 codewords)
    while len(codewords) < 19:
        if len(codewords) % 2 == 0:
            codewords.append(0xEC)
        else:
            codewords.append(0x11)
    return codewords

def qr_decode_byte_mode(data: list[int]) -> str:
    """Decode a list of byte values from 1-L qr code byte mode to a string."""
    # Convert integers back to bitstream
    bitstream = ''.join(format(byte, '08b') for byte in data)

    # Read mode indicator (first 4 bits)
    mode_indicator = bitstream[:4]
    if mode_indicator != "0100":
        raise ValueError("Unsupported mode indicator")

    # Read character count (next 8 bits for version 1 Byte mode)
    char_count = int(bitstream[4:12], 2)

    # Read the characters
    chars = []
    for i in range(char_count):
        byte_value = int(bitstream[12 + i*8:20 + i*8], 2)
        chars.append(chr(byte_value))

    return ''.join(chars)

print(qr_encode_byte_mode("HELLO"))

print(qr_decode_byte_mode(qr_encode_byte_mode("HELLO")))
list1 = [64, 84, 132, 84, 196, 196, 240, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236]
list2 = [0, 84, 132, 84, 196, 196, 240, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236] #will not work as first byte is wrong its decide the mode
list3 = [64, 0, 132, 84, 196, 196, 240, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236] #will not give correct result as second byte is wrong
print(qr_decode_byte_mode(list1))
print(qr_decode_byte_mode(list3))
#so the point is to make list 3 work even when it has one wrong byte using error correction
#i need to implement reed solomon error correction for that