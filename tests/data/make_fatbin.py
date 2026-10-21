#!/usr/bin/env python3
"""Builds the test fat binaries from vector_add.ptx (needs the `lz4` Python module).

    python3 make_fatbin.py        writes vector_add_lz4.fatbin and vector_add_plain.fatbin

Layout (little-endian), as read by src/ptx/fatbin.cpp:
    header  u32 magic 0xBA55ED50, u16 version 1, u16 header size 16, u64 size of all entries
    entry   u16 kind (1 PTX, 2 ELF), u16 version, u32 header size (64), u32 padded payload size,
            u32 0, u32 payload size, u32 0, u32 0, u32 sm version, u32 address bits, u32 0,
            u64 flags (0x1 64-bit, 0x10 Linux, 0x2000 LZ4), u64 0, u64 uncompressed size
"""
import os
import struct

import lz4.block

HERE = os.path.dirname(os.path.abspath(__file__))
FLAG_64BIT, FLAG_LINUX, FLAG_LZ4 = 0x1, 0x10, 0x2000


def entry(kind, sm, payload, flags=FLAG_64BIT | FLAG_LINUX, uncompressed=0):
    padded = payload + b"\0" * (-len(payload) % 8)
    header = struct.pack("<HHIIIIIIIIIQQQ", kind, 0x0101, 64, len(padded), 0, len(payload), 0, 0, sm, 64, 0,
                         flags, 0, uncompressed)
    assert len(header) == 64
    return header + padded


def fatbin(entries):
    body = b"".join(entries)
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def main():
    with open(os.path.join(HERE, "vector_add.ptx"), "rb") as f:
        ptx = f.read() + b"\0"
    # A newer PTX that must not be picked for the default compute capability 7.5.
    newer = (b".version 8.0\n.target sm_90\n.address_size 64\n"
             b".visible .entry vecAdd(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n)\n"
             b"{ .reg .b32 %r<2>; vulkanizer_wrong_ptx_selected.b32 %r1, 0; ret; }\n\0")
    cubin = b"\x7fELF" + b"\0" * 60

    compressed = lz4.block.compress(ptx, store_size=False, mode="high_compression")
    files = {
        "vector_add_lz4.fatbin": fatbin([
            entry(2, 75, cubin),
            entry(1, 52, compressed, FLAG_64BIT | FLAG_LINUX | FLAG_LZ4, len(ptx)),
            entry(1, 90, newer),
        ]),
        "vector_add_plain.fatbin": fatbin([entry(1, 90, newer), entry(1, 52, ptx)]),
        "cubin_only.fatbin": fatbin([entry(2, 75, cubin), entry(2, 86, cubin)]),
    }
    for name, data in files.items():
        with open(os.path.join(HERE, name), "wb") as f:
            f.write(data)
        print(f"{name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
