"""
Experiment 6 - Image Compression using Huffman Coding
-----------------------------------------------------
Goal: losslessly compress a grayscale image by giving frequent pixel values
SHORT bit codes and rare ones LONG codes, then report the savings.
Input : exp6_input.png
Output: printed code table + compression statistics (no file is written)

Steps: count pixel frequencies -> build Huffman tree -> read codes off the tree
       -> compare original vs compressed size.
"""
import cv2
import numpy as np
import heapq
import os


class Node:
    """One node of the Huffman tree (leaf = a pixel value, internal = a merge)."""

    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq      # how often this symbol (or whole subtree) occurs
        self.symbol = symbol  # pixel value (None for internal nodes)
        self.left = left
        self.right = right
        self.huff = ''        # bit on the edge to this node: '0' (left) or '1' (right)

    # heapq needs this to always pop the node with the LOWEST frequency
    def __lt__(self, nxt):
        return self.freq < nxt.freq


def main():
    image_name = 'exp6_input.png'

    if not os.path.exists(image_name):
        print(f"Error: {image_name} not found in the current directory.")
        return

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # --- Step 1: count how many times each pixel value (0-255) appears ---
    frequencies = {}
    for row in img:
        for pixel in row:
            if pixel in frequencies:
                frequencies[pixel] += 1
            else:
                frequencies[pixel] = 1

    # --- Step 2: build the Huffman tree using a min-heap ---
    # Start with one leaf per symbol
    nodes = []
    for symbol, freq in frequencies.items():
        heapq.heappush(nodes, Node(freq, symbol))

    # Repeatedly merge the two rarest nodes into a parent until one root remains
    while len(nodes) > 1:
        left = heapq.heappop(nodes)    # rarest
        right = heapq.heappop(nodes)   # second rarest

        left.huff = '0'
        right.huff = '1'

        newNode = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(nodes, newNode)

    # --- Step 3: walk the tree; the bits along the path to a leaf are its code ---
    huffman_codes = {}

    def generate_codes(node, val=''):
        newVal = val + str(node.huff)
        if node.left:
            generate_codes(node.left, newVal)
        if node.right:
            generate_codes(node.right, newVal)
        # Leaf node -> store the finished code for its pixel value
        if not node.left and not node.right:
            huffman_codes[node.symbol] = newVal

    generate_codes(nodes[0])

    # --- Step 4: measure compression ---
    original_size_bits = height * width * 8   # raw 8-bit grayscale
    original_bytes = height * width

    # Encoded data: each symbol's frequency x its code length
    compressed_size_bits = sum(frequencies[symbol] * len(huffman_codes[symbol]) for symbol in frequencies)
    # Overhead to store the code table (8 bits for symbol + its code), so the file is decodable
    metadata_size_bits = sum(8 + len(huffman_codes[symbol]) for symbol in frequencies)
    total_compressed_bits = compressed_size_bits + metadata_size_bits

    compression_ratio = original_size_bits / total_compressed_bits           # >1 means smaller
    space_saving = (1 - (total_compressed_bits / original_size_bits)) * 100  # percent saved

    # --- Report ---
    print(f"Original Image: {image_name}\n")
    print(f"Image Dimensions: {width} x {height}")
    print(f"Number of Pixel Intensities: {len(frequencies)}")
    print(f"Original Image Size: {original_bytes} bytes\n")

    print("Huffman Coding Table:")
    print(f"{'Pixel Intensity':<20} {'Frequency':<15} {'Huffman Code'}")

    for symbol in sorted(frequencies.keys()):
        print(f"{symbol:<20} {frequencies[symbol]:<15} {huffman_codes[symbol]}")

    print(f"\nOriginal Size:       {original_size_bits} bits")
    print(f"Compressed Size:     {total_compressed_bits} bits")
    print(f"Compression Ratio:   {compression_ratio:.4f}")
    print(f"Space Saving:        {space_saving:.2f} %\n")
    print("Image compression using Huffman Coding completed successfully.")


if __name__ == "__main__":
    main()
