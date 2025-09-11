import string
import matplotlib.pyplot as plt
#pip install matplotlib
import numpy as np

print("will try with qr")
def qr_encode_byte_mode(data: str) -> list[int]: 
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

    # Pad with 0xEC and 0x11 to reach the required length for  (19 codewords)
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

# Generator polynomial for 7 error correction codewords 
def _generate_generator_poly(degree: int) -> list[int]:
    """Generate the generator polynomial for Reed-Solomon encoding."""
    g = [1]
    for i in range(degree):
        g = _poly_mul(g, [1, EXP_TABLE[i]])# Multiply by (x - α^i)
    return g

errorcorrection_poly = _generate_generator_poly(7)  # 7 error correction codewords for 

def ReedSolomon_encode(data19: list[int]) -> list[int]:
    """Encode data using Reed-Solomon error correction """
    if len(data19) != 19:
        raise ValueError("Data must be exactly 19 bytes for ")

    ecc = [0] * 7  # 7 error correction codewords for 
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
    """Decode data using Reed-Solomon error correction."""
    if len(data26) != 26:
        raise ValueError("Data must be exactly 26 bytes")

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

#QR placment
def finderPattern(top: int, left: int, matrix: list[list[int]]):
    """Place a finder pattern at the specified position in the matrix."""
    pattern = [
        [5, 5, 5, 5, 5, 5, 5],
        [5, 4, 4, 4, 4, 4, 5],
        [5, 4, 5, 5, 5, 4, 5],
        [5, 4, 5, 5, 5, 4, 5],
        [5, 4, 5, 5, 5, 4, 5],
        [5, 4, 4, 4, 4, 4, 5],
        [5, 5, 5, 5, 5, 5, 5],
    ]
    size = len(matrix)
    for r in range(-1, 8):   # -1..7 → separator + 7x7 core + separator
        for c in range(-1, 8):
            rr, cc = top + r, left + c
            if 0 <= rr < size and 0 <= cc < size:
                if 0 <= r < 7 and 0 <= c < 7:
                    matrix[rr][cc] = pattern[r][c]
                else:
                    matrix[rr][cc] = 4  # separator

def alignmentPattern(top: int, left: int, matrix: list[list[int]]):
    """Place an alignment pattern at the specified position in the matrix."""
    pattern = [
        [5, 5, 5, 5, 5],
        [5, 4, 4, 4, 5],
        [5, 4, 5, 4, 5],
        [5, 4, 4, 4, 5],
        [5, 5, 5, 5, 5],
    ]
    for r in range(5):
        for c in range(5):
            matrix[top + r][left + c] = pattern[r][c]

def twentyfiveby25matrix() -> list[list[int]]:
    """Create a 25x25 matrix initialized to -1 (unset)."""
    return [[-1 for _ in range(25)] for _ in range(25)]
    
def timePattern(matrix: list[list[int]]):
    """Place timing patterns in the QR code matrix."""
    n = len(matrix)
    for i in range(8, n - 8):
        # Horizontal timing pattern
        matrix[6][i] = 5 if i % 2 == 0 else 4
        # Vertical timing pattern
        matrix[i][6] = 5 if i % 2 == 0 else 4
def reserve_format_info_areas(matrix: list[list[int]]):
    """Reserve format information areas in the QR code matrix with placeholders (-2)."""
    n = len(matrix)

    # Top-left: row 8 and col 8
    for i in range(9):
        if i != 6:  # skip timing line intersection
            if matrix[8][i] == -1:   # row 8
                matrix[8][i] = -2
            if matrix[i][8] == -1:   # col 8
                matrix[i][8] = -2

    # Top-right: row 8, last 8 columns
    for i in range(8):
        if matrix[8][n - 1 - i] == -1:
            matrix[8][n - 1 - i] = -2

    # Bottom-left: col 8, last 8 rows
    for i in range(8):
        if matrix[n - 1 - i][8] == -1:
            matrix[n - 1 - i][8] = -2

    # Dark module (always black)
    matrix[n - 8][8] = 5

def place_patterns(matrix: list[list[int]]):
    """Place finder and alignment patterns in the QR code matrix."""
    n = len(matrix)
    edge = n - 7  
    # Place finder patterns
    finderPattern(0, 0, matrix)          # Top-left (need to put it outside the border)
    finderPattern(0, edge, matrix)         # Top-right
    finderPattern(edge, 0, matrix)         # Bottom-left

    # Place alignment pattern (only one at (18,18))
    alignmentPattern(16, 16, matrix)
    # Place timing patterns
    timePattern(matrix)
    # Reserve format information areas
    reserve_format_info_areas(matrix)

def print_matrix(matrix: list[list[int]]):
    """Print the QR code matrix."""
    for row in matrix:
        print(' '.join(
            '#' if cell == 1 else '.' if cell == 0 else '?' if cell == -2 else '#' if cell == 1 else '.' if cell == 4 else '#' if cell == 5 else' '
            for cell in row
        ))

print("looking at placement")
matrix = twentyfiveby25matrix()
place_patterns(matrix)
print_matrix(matrix)

def place_data(matrix: list[list[int]], data: list[int]):
    """Place data bits into the QR code matrix."""
    n = len(matrix)
    bitstream = ''.join(format(byte, '08b') for byte in data)  # Convert data to bitstream
    bit_index = 0

    # Start from the bottom-right corner
    col = n - 1
    row = n - 1
    direction = -1  # -1 for up, 1 for down

    while col > 0:
        if col == 6:  # Skip vertical timing pattern
            col -= 1

        for i in range(n):
            r = row + direction * i
            for c in [col, col - 1]:  # Right column first, then left
                if matrix[r][c] == -1:  # Only place in unset areas
                    if bit_index < len(bitstream):
                        matrix[r][c] = int(bitstream[bit_index])
                        bit_index += 1
                    else:
                        matrix[r][c] = 0  # Pad with white if no more data

        row += direction * (n - 1)
        direction *= -1  # Change direction
        col -= 2  # Move to the next pair of columns
    return matrix
print("looking at placement with data")
data_with_ecc = ReedSolomon_encode(qr_encode_byte_mode("HELLO"))    
matrix_with_data = place_data(matrix, data_with_ecc)
print_matrix(matrix_with_data)

#now we have a qr code matrix with data placed in it
#mask it but need to considcer not to mask the patterns
def apply_mask(matrix: list[list[int]], mask_pattern: int) -> list[list[int]]:
    """Apply a mask pattern to the QR code matrix but avoid the pattenrs."""
    n = len(matrix)
    for r in range(n):
        for c in range(n):
            if matrix[r][c] in (0, 1):  # Only apply mask to data areas
                if mask_pattern == 0 and (r + c) % 2 == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 1 and r % 2 == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 2 and c % 3 == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 3 and (r + c) % 3 == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 4 and (r // 2 + c // 3) % 2 == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 5 and ((r * c) % 2 + (r * c) % 3) == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 6 and (((r * c) % 2 + (r * c) % 3) % 2) == 0:
                    matrix[r][c] ^= 1
                elif mask_pattern == 7 and (((r + c) % 2 + (r * c) % 3) % 2) == 0:
                    matrix[r][c] ^= 1
    return matrix
    
apply_mask(matrix_with_data, 0)
print("looking at placement with data and mask")
print()
print_matrix(matrix_with_data)

def place_format_info(matrix, ec_level='L', mask=0):
    # Format strings for EC=L and mask=0 (precomputed)
    format_str = "111011111000100"
    n = len(matrix)

    # Top-left around finder
    for i in range(6):
        matrix[i][8] = int(format_str[i])  # col 8, rows 0–5
    matrix[7][8] = int(format_str[6])
    matrix[8][8] = int(format_str[7])
    matrix[8][7] = int(format_str[8])
    for i in range(6):
        matrix[8][5 - i] = int(format_str[9 + i])

    # Top-right
    for i in range(8):
        matrix[8][n - 1 - i] = int(format_str[i])

    # Bottom-left
    for i in range(7):
        matrix[n - 1 - i][8] = int(format_str[8 + i])

place_format_info(matrix_with_data)
print("looking at placement with data and mask and format info")
print()
print_matrix(matrix_with_data)

def matrix_to_image(matrix: list[list[int]], filename="qr_matrix.png", scale=20):
    """
    Render a QR matrix into an image and save it.
    
    - 1 -> black
    - 0 -> white
    - 4 -> white (reserveed bits for patterns)
    - 5 -> black (reserved bits for patterns)
    """
    n = len(matrix)

    # Map values to colors
    img = []
    for row in matrix:
        img_row = []
        for cell in row:
            if cell == 1:
                img_row.append(0)      # black
            elif cell == 0:
                img_row.append(1)      # white
            elif cell == 5:
                img_row.append(0)    # reserved: black
            else:
                img_row.append(1)      # treat rest as white
        img.append(img_row)

    # Plot
    plt.figure(figsize=(n*scale/100, n*scale/100))
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"QR matrix saved as {filename}")
matrix_to_image(matrix_with_data)
def image_to_matrix(filename="qr_matrix.png", modules=25) -> list[list[int]]:
    """Convert an image back to a QR code logical matrix (e.g. 25x25)."""
    img = plt.imread(filename)
    if img.ndim == 3:
        img = img[:, :, 0]  # grayscale

    # Normalize (0=black, 1=white)
    img = (img > 0.5).astype(int)

    n_pixels = img.shape[0]
    module_size = n_pixels // modules  # how many pixels per module

    matrix = [[-1 for _ in range(modules)] for _ in range(modules)]

    for r in range(modules):
        for c in range(modules):
            block = img[r*module_size:(r+1)*module_size,
                        c*module_size:(c+1)*module_size]
            # Majority vote inside block
            matrix[r][c] = 1 if np.mean(block) < 0.5 else 0

    return matrix
read_matrix = image_to_matrix("qr_matrix.png")
print("looking at read matrix from image")
print_matrix(read_matrix)
