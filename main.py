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
#I need to implement reed solomon error correction for that but first byte mode cant be wrong as it decide the mode

#so lets implement reed solomon error correction
# to do that i first need to implement galois field arithmetic this defines the field GF(2^8) used in QR codes i have 256 elements 0-255 
def _build_galois_tables():
    """Build the Galois field tables for GF(2^8)."""
    global EXP_TABLE, LOG_TABLE
    EXP_TABLE = [0] * 512  # Extended to 512 for easy multiplication
    LOG_TABLE = [0] * 256

    x = 1
    for i in range(255):
        EXP_TABLE[i] = x
        LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:  # If x is 256 or more, reduce it
            x ^= 0x11d  # Primitive polynomial

    for i in range(255, 512):
        EXP_TABLE[i] = EXP_TABLE[i - 255]
    return EXP_TABLE, LOG_TABLE

def _galois_mul(x: int, y: int) -> int:
    """Multiply two numbers in GF(2^8)."""
    if x == 0 or y == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[x] + LOG_TABLE[y]]
def _poly_mul(p: list[int], q: list[int]) -> list[int]:
    """Multiply two polynomials in GF(2^8)."""
    result = [0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            result[i + j] ^= _galois_mul(p[i], q[j])
    return result
def _galois_inv(x: int) -> int:
    """Find the multiplicative inverse in GF(2^8)."""
    if x == 0:
        raise ZeroDivisionError("Cannot invert zero in a Galois field")
    return EXP_TABLE[255 - LOG_TABLE[x]]

_PRIM = 0x11d # Primitive polynomial for GF(2^8)
EXP_TABLE, LOG_TABLE = _build_galois_tables() # Initialize Galois field tables

# Generator polynomial for 7 error correction codewords (version 1-L)
def _generate_generator_poly(degree: int) -> list[int]:
    """Generate the generator polynomial for Reed-Solomon encoding."""
    g = [1]
    for i in range(degree):
        g = _poly_mul(g, [1, EXP_TABLE[i]])# Multiply by (x - α^i)
    return g

errorcorrection_poly = _generate_generator_poly(7)  # 7 error correction codewords for version 1-L

def ReedSolomon_encode(data19: list[int]) -> list[int]:
    """Encode data using Reed-Solomon error correction (for QR code version 1-L)."""
    if len(data19) != 19:
        raise ValueError("Data must be exactly 19 bytes for version 1-L")

    ecc = [0] * 7  # 7 error correction codewords for version 1-L
    for byte in data19:
        factor = byte ^ ecc[0]
        # Shift left
        ecc = ecc[1:] + [0]
        for i in range(7):
            ecc[i] ^= _galois_mul(factor, errorcorrection_poly[i + 1])

    return data19 + ecc  # Return data + error correction codewords

encodeval = qr_encode_byte_mode("HELLO")
print(ReedSolomon_encode(encodeval))
#now we can see that we added 7 error correction codewords to the original 19 codewords
#now we need to implement reed solomon decode with error correction will only correct 1 byte error otherwise i need to implement more complex algorithm
def ReedSolomon_decode(data26: list[int]) -> list[int]:
    """Decode data using Reed-Solomon error correction (for QR code version 1-L)."""
    if len(data26) != 26:
        raise ValueError("Data must be exactly 26 bytes for version 1-L")

    # Calculate syndromes
    syndromes = []
    error = False
    for i in range(7):
        x = EXP_TABLE[i] # α^(i)
        s = 0 
        for j in range(26):
            s = _galois_mul(s, x) ^ data26[j]
        syndromes.append(s)

    if max(syndromes) == 0:
        return data26[:19]  # No errors detected

    # For simplicity, we will only correct single errors will need more complex algorithm for multiple errors
    S0=syndromes[0] # is the error value
    #we can now see if we have error it will now return empty list if one is wrong
    r = _galois_mul(syndromes[1], _galois_inv(S0)) #position of error
    pos_pow = LOG_TABLE[r]           # this is (n-1-pos)
    n = len(data26)
    pos = (n - 1) - pos_pow
     # Apply correction: magnitude = S0
    corrected = data26[:]
    corrected[pos] ^= S0
    check = []
    for i in range(7):
        x = EXP_TABLE[i]
        s = 0
        for j in range(26):
            s = _galois_mul(s, x) ^ corrected[j]
        check.append(s)
    if max(check) != 0:
        raise ValueError("Unable to correct (likely >1 error).")

    return corrected[:19]

print("looking at decode using reed solomon")
print(ReedSolomon_decode(ReedSolomon_encode(encodeval))) #should return the original 19 codewords
fakevalue = [64, 0, 132, 84, 196, 196, 240, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 95, 105, 74, 31, 73, 149, 139]       
print(ReedSolomon_decode(fakevalue)) #dont gives error seems to correct lets try diffrent cases
fakevalue2 = [64, 84, 0, 84, 196, 196, 240, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 95, 105, 74, 31, 73, 149, 139]       
print(ReedSolomon_decode(fakevalue2))
#seems to work for single byte errors 

qrdata = qr_decode_byte_mode(ReedSolomon_decode(fakevalue2)) 
print(qrdata) #should return HELLO

#need to look at QR documentation to implement the placment of codewords in the QR matrix