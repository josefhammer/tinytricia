#
# TinyTricia
#
# A space-optimized Patricia Trie for storing keys with up to 256 bits (optionally: key/value pairs).
#
# (c) 2022 Josef Hammer (josefhammer.com)
#
# Source: https://github.com/josefhammer/tinytricia
#

from distutils.core import setup

setup(
    name="TinyTricia",
    version="1.0.0",
    py_modules=['TinyTricia'],
    description="A space-optimized Patricia Trie for storing keys with up to 256 bits (optionally: key/value pairs).",
    author="Josef Hammer",
    author_email="josef@hammer.design",
    url="https://github.com/josefhammer/tinytricia",
    keywords=['patricia tree', 'IP addresses', 'IPv4', 'IPv6'],
    long_description='''
    A space-optimized Patricia Trie for storing keys with up to 256 bits (optionally: key/value pairs).

    __Design goal__ 
    Low memory consumption.
    
    As an example, for 16.78 million (2^24) 48-bit keys (IPv4 + port number), 
    TinyTricia requires only between 134 MB (best case) and 268 MB (worst case) of memory space.

    __Features__
    * Storage of values can be disabled using the `keysOnly` parameter (`__init__`).
    * Nodes do not contain a link to their parent.
    * An optimized __Compact Mode__ is activated if keys are <= 57 bits _and_ no values are stored.
    * __Compact Mode__: Max 536,870,912 keys (2^29) in the best case; half in the worst case.
    * __Regular Mode__: Max 268,435,455 keys (2^28-1) in the best case; half in the worst case.
    * _Best case_: For each key there is another key where only the rightmost bit differs.
    * _Worst case_: For no key there is another key where only the rightmost bit differs.
    
    __Limitations__
    Only keys with a fixed length are stored, thus only leaf nodes will contain a full key. 
    Ints are used for the keys, thus the algorithm will require a little-endian machine 
    to work correctly.
    ''')
