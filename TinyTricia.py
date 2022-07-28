#
# TinyTricia
#
# A space-optimized Patricia Trie for storing keys with up to 256 bits (optionally: key/value pairs).
#
# (c) 2022 Josef Hammer (josefhammer.com)
#
# Source: https://github.com/josefhammer/tinytricia
#

from __future__ import annotations
from array import array
from collections import deque
import pickle


class TinyTricia(object):
    """
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

    """
    # __Internal data structure__
    #
    # __bucketSize__ how many bits of the key go into one bucket
    # __numBuckets__ number of buckets needed to store all bits of one key
    # __id__         unique ID for each key/data pair (range 1..2^28-1)
    #
    # __nodes__ node[]
    # __node__  int64 -- prefix (bits 56..63 == 8 bits), rightChildIndex (28..55), leftChildIndex (0..27)
    #
    # __keys__  int(bucketSize)[] -- access via id*numBuckets
    # __data__  python list -- access via id

    # __Compact Mode__
    #
    # Keys are stored directly in the leaf nodes. The prefix is reduced to 6 bits -> both child indices are
    # increased to 29 bits. Least significant bit (LSB) of the key is removed/added on insertion/read.
    #
    # __leafNode__ int64 -- prefix (58..63), rightKeyAvail (57), leftKeyAvail (56), key(0..55) with LSB removed
    #
    __slots__ = [
        "_head", "_nodes", "_keys", "_numKeys", "NUM_BUCKETS", "BUCKET_SIZE", "MAX_PREFIX", "KEYS_ONLY", "COMPACT_MODE",
        "PREFIX_SHIFT", "INDEX_SHIFT", "INDEX_MASK", "_data", "_id", "key", "keysMatch"
    ]

    def __init__(self, numBits=32, keysOnly=False, minBucketSize=16):

        assert (numBits <= 256)  # 8 bits space for the prefix only

        self._initConsts(numBits, keysOnly, minBucketSize)

        self._nodes = self._createArrayWithBitsPerItem(64)  # uint64
        self._nodes.append(0)  # fake root node to speed up loops and allow nullptr
        self._head = 0  # position of initial root element
        self._numKeys = 0
        self._keys = None
        self._data = None

        if not self.COMPACT_MODE:
            self._keys = self._createArrayWithBitsPerItem(self.BUCKET_SIZE)
            self._data = []  # pointers to the data objects
            self._addKey(0, None)  # key/data no. zero == key/data not available (nullptr)
            self._numKeys = 0  # set back to zero after dummy key no. zero (prev line)

    def _initConsts(self, numBits, keysOnly, minBucketSize):

        self.MAX_PREFIX = numBits
        self.KEYS_ONLY = keysOnly
        self.COMPACT_MODE = numBits <= 57 and keysOnly

        # reduce prefix by 2 bits in COMPACT_MODE (prefix is <= 57)
        self.PREFIX_SHIFT = 58 if self.COMPACT_MODE else 56
        self.INDEX_SHIFT = 29 if self.COMPACT_MODE else 28
        self.INDEX_MASK = 0x1FFFFFFF if self.COMPACT_MODE else 0xFFFFFFF

        self.NUM_BUCKETS, self.BUCKET_SIZE = self._calcBucketSize(numBits, minBucketSize)

        if self.COMPACT_MODE:
            self._id = self._idCompact
            self.key = self._keyCompact
            self.keysMatch = self._keysMatchCompact
        else:
            self._id = self._idRegular
            self.key = self._keyRegular
            self.keysMatch = self._keysMatchRegular

    def numNodes(self):
        return len(self._nodes) - 1

    def numKeys(self):
        return self._numKeys  # without COMPACT_MODE: (len(self._keys) // self.NUM_BUCKETS) - 1

    def contains(self, key):
        return self.search(key)[3]  # [3] ... isFound

    def containsFirstNBits(self, key) -> tuple[int, list[int]]:
        """
        Returns (n, prefixes); (0,[]) if tree is empty.
        
        :param n: The closest key shares the first `n` bits with the given `key`.
        :param prefixes: The parent prefixes at which the closest key is attached (until incl. prefix `n`).
        """

        if not self._head:  # diffPos == 0 --> would return MAX_PREFIX
            return 0, []
        parents, _, _, diffPos = self._diffPos(key)

        firstN = (self.MAX_PREFIX - diffPos - 1)  # diffPos needs to be fixed by 1 (see _diffPos())

        # Calculate parent prefixes (and remove first parent (fake 0)).
        # NOTE: Can't slice a deque, thus easier to remove first parent _after_ the list comprehension.
        #
        prefixes = [self.MAX_PREFIX - (self._nodes[parent] >> self.PREFIX_SHIFT) for parent in parents][1:]
        prefixes = [prefix for prefix in prefixes if prefix <= firstN]  # remove all entries > firstN

        return firstN, prefixes

    def add(self, key):
        """
        Convencience wrapper around set(). A more sensible name if only keys are stored.
        """
        return self.set(key)

    def get(self, key):
        """
        Returns (None, None) if key not found.
        Returns (key, value) if key is found.
        """
        _, _, id, isFound = self.search(key)
        return (None, None) if not isFound else (key, self.getData(id))

    def set(self, key, data=None):
        """
        Adds or sets a key and the given data value.
        """
        return self._set(key, data)

    def save(self, filename):
        """
        Saves the current state / Patricia Tree to a file.
        """
        with open(filename, 'wb') as handle:

            pickle.dump("v1", handle, protocol=pickle.HIGHEST_PROTOCOL)
            for item in [
                    self.MAX_PREFIX, self.KEYS_ONLY, self.BUCKET_SIZE, self._head, self._numKeys, self._nodes,
                    self._keys, self._data
            ]:
                pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Loads a previously saved state of this object from a file.
        """
        with open(filename, 'rb') as handle:

            version = pickle.load(handle)
            assert (version == "v1")

            MAX_PREFIX = pickle.load(handle)
            KEYS_ONLY = pickle.load(handle)
            bucketSize = pickle.load(handle)
            self._initConsts(MAX_PREFIX, KEYS_ONLY, bucketSize)

            self._head = pickle.load(handle)
            self._numKeys = pickle.load(handle)
            self._nodes = pickle.load(handle)
            self._keys = pickle.load(handle)
            self._data = pickle.load(handle)

    def __len__(self):
        return self.numKeys()

    def __iter__(self):
        """ 
        Generator yields (nodeID, prefix, value) tuples.

        If prefix = 0: value is a key.
        If prefix > 0: value is a (leftNodeID, rightNodeID) tuple.

        See _iter().
        """
        for elt in self._iter(self._head):
            yield elt

    ### Semi-private methods below: These allow additional functionality if necessary ##################################

    def getData(self, id):
        return None if self.KEYS_ONLY else self._data[id]

    def setData(self, id, value):
        if id and not self.KEYS_ONLY:
            self._data[id] = value

    def search(self, key):
        """
        Search for key. 
        Returns (parentNodesIndices:deque, nodeIndex:int, id:int, isFound:bool) tuple.
        """
        parents = deque([0])  # root parent points to empty node 0 (i.e. prefix will also be 0)
        pos = self._head
        nodes = self._nodes  # avoid object lookup
        prefixShift = self.PREFIX_SHIFT
        indexShift = self.INDEX_SHIFT
        indexMask = self.INDEX_MASK

        nodeVal = nodes[pos]
        prefix = nodeVal >> prefixShift

        while prefix:
            parents.append(pos)
            #                 bitAt() inlined
            pos = ((nodeVal >> (((key >> prefix) & 0x1) * indexShift)) & indexMask)
            nodeVal = nodes[pos]
            prefix = nodeVal >> prefixShift

        id = self._id(nodeVal, key & 0x1)

        # If we hit a nullptr (pos == 0, e.g. empty root), we did not find anything and id will be 0.
        #
        return parents, pos, id, id and self.keysMatch(key, id)  # parent, index, id, isFound

    ### Private methods below ##########################################################################################

    def _calcBucketSize(self, numBits, minBucketSize):
        """
        Calculate the ideal size of a key bucket, i.e., the biggest possible bucket (max 64 bits) 
        that does not waste any memory space (unless enforced by `minBucketSize`).
        Rationale: Faster access with fewer but bigger buckets (instead of using 8 by default).

        E.g., minBucketSize = 8 bits:

        *) 48 bits => bucketSize=16, numBuckets=3:
           (48+7)//8 = 6 -> (48+15)//16 = 3

        *) 64 bits => bucketSize=64, numBuckets=1:
           (64+7)//8 = 8 -> (64+15)//16 = 4 -> (64+31)//32 = 2 -> (64+63)//64 = 1
        """

        bucketSize = minBucketSize
        numBuckets = (numBits + bucketSize - 1) // bucketSize  # ceiling division

        while bucketSize < 64 and numBuckets % 2 == 0:  # only if even number (don't waste memory space)
            bucketSize <<= 1
            numBuckets >>= 1
        return numBuckets, bucketSize

    def _createArrayWithBitsPerItem(self, bits):
        """
        Creates a Python array with a base type that can hold the the requested number of bits (e.g., 64 bit integer).
        """

        for type in ['B', 'H', 'I', 'L', 'Q']:
            if array(type).itemsize * 8 >= bits:
                return array(type)
        assert (False)

    def _idCompact(self, nodeVal, isRight):
        #
        # Returns key (incl. LSB) and the two keyAvailable bits in COMPACT_MODE.
        #
        # The two keyAvailable bits need to be included to make sure the ID is >0
        # even if the key itself is 0 (to be able to distinguish "key=0" from "no key available"
        # (see search()).
        #
        if (nodeVal >> 56 >> isRight) & 0x1:  # key is available
            return ((nodeVal & 0x3FFFFFFFFFFFFFF) << 1) + isRight  # add lowest bit again

        return 0  # no key available

    def _idRegular(self, nodeVal, isRight):
        return (nodeVal >> (isRight * 28)) & 0xFFFFFFF

    def _keyCompact(self, id):
        #
        # in COMPACT_MODE the ID is the key (incl. LSB) and including the two keyAvailable bits
        #
        return id & 0x1FFFFFFFFFFFFFF  # mask the full 57 bits of the key (includes LSB)

    def _keyRegular(self, id):
        #
        # put the key together from its buckets
        #
        result = 0
        id *= self.NUM_BUCKETS  # calculcate position of first bucket
        for i in range(0, self.NUM_BUCKETS):
            result = (result << self.BUCKET_SIZE) + self._keys[id + i]
        return result

    def _keysMatchCompact(self, key, id):
        """
        Checks if the given `key` matches the one identified by `id`.
        See _keyCompact().
        """
        return key == (id & 0x1FFFFFFFFFFFFFF)

    def _keysMatchRegular(self, key, id):
        """
        Checks if the given `key` matches the one identified by `id`.
        Avoids assembling the whole key, but checks only as many buckets as necessary.
        """
        bitmask = (1 << (self.BUCKET_SIZE)) - 1  # bitmask for a single bucket
        id *= self.NUM_BUCKETS  # calculcate position of first bucket

        for i in range(self.NUM_BUCKETS - 1, -1, -1):
            if self._keys[id + i] != (key & bitmask):
                return False
            key >>= self.BUCKET_SIZE
        return True

    def _addKey(self, key, data):

        self._numKeys += 1

        if self.COMPACT_MODE:
            #
            # just return the key as the ID
            #
            return key

        bitmask = (1 << (self.BUCKET_SIZE)) - 1  # bitmask for a single bucket
        id = len(self._keys) // self.NUM_BUCKETS
        assert (id <= 0xFFFFFFF)  # must fit into 28 bits

        # add key parts according to bucket size (most significant bit in the first bucket)
        #
        for i in range(self.NUM_BUCKETS - 1, -1, -1):
            self._keys.append((key >> (i * self.BUCKET_SIZE)) & bitmask)  # first bucket may contain 'empty' bits

        if not self.KEYS_ONLY:
            self._data.append(data)
        return id

    def _addNode(self, prefix, childIsRight, childIndex):

        nodeIndex = len(self._nodes)
        assert (nodeIndex <= self.INDEX_MASK)  # must fit into INDEX_MASK

        if not prefix and self.COMPACT_MODE:
            nodeVal = self._compactKeyAvailBit(childIsRight) | (childIndex >> 1)  # remove lowest bit
        else:
            nodeVal = (childIndex << (childIsRight * self.INDEX_SHIFT))

        self._nodes.append((prefix << self.PREFIX_SHIFT) | nodeVal)
        return nodeIndex

    def _bitAt(self, key, pos):
        """ Returns the bit at position <pos> for <pos> in (0,n]. """

        return (key >> pos) & 0x1  # pos is guaranteed to be >= 0

    def _compactKeyAvailBit(self, isRight):
        return 1 << 56 << isRight

    def _calcDiffPos(self, key, otherKey):
        """ Finds the position [1,n] of the most significant _different_ bit (bit 1 == least significant bit).
            Returns 0 if both keys are equal.
        """

        # Find all different bits
        #
        diff = key ^ otherKey

        # Shift right until diff is zero (i.e. we got the first different bit)
        #
        pos = 0
        while (diff):
            diff >>= 1
            pos += 1
        return pos

    def _diffPos(self, key):
        """
        Returns (parents, node, id, diffPos), with diffPos being the prefix at which the key should be inserted into the Patricia
        trie. Returns -1 for diffPos if `key` exists in the tree (exact match).
        """

        # Find the location where the item should be inserted
        #
        (parents, node, id, isFound) = self.search(key)

        if isFound:
            diffPos = -1
        elif not self._head:
            diffPos = 0  # prefix 0 if empty tree
        else:
            if not id:
                id = self._id(self._nodes[node], 1 - (key & 0x1))  # maybe on the peer? One of both must have an ID.
            diffPos = self._calcDiffPos(key, self.key(id)) - 1

        return (parents, node, id, diffPos)

    def _set(self, key, data=None):

        # Is key in the allowed bit range? (no overflow)
        #
        assert (key is not None and key >= 0 and not (key >> self.MAX_PREFIX))

        # Find the location where it should be inserted into the trie
        #
        (parents, node, id, diffPos) = self._diffPos(key)

        if diffPos < 0:  # exists already
            self.setData(id, data)
            return node

        nodes = self._nodes

        # Does not exist yet --> add new key to array
        #
        newID = self._addKey(key, data)

        # Are we on the key node level (prefix == 0)?
        #
        if not diffPos:  # diffPos 0: two IDs are stored in a single node to save space
            nodeVal = nodes[node]
            if not (nodeVal >> self.PREFIX_SHIFT):
                otherKeyIsRight = 1 - (key & 0x1)
                otherKey = self.key(self._id(nodeVal, otherKeyIsRight))
                if (key ^ otherKey) == 1:  # only last bit different, i.e. we belong into the peer slot
                    if self.COMPACT_MODE:
                        #
                        # if COMPACT_MODE: set available-bit only, the key is already there
                        #
                        self._nodes[node] |= self._compactKeyAvailBit(key & 0x1)
                    else:
                        self._setChildIndex(node, key & 0x1, newID)
                    return node

        # Otherwise backtrack: Where shall we insert it?
        #
        parent = parents.pop()
        while parent and (nodes[parent] >> self.PREFIX_SHIFT) <= diffPos:
            node, parent = parent, parents.pop()

        newNode = self._addNode(0, key & 0x1, newID)

        # Found a node with the required prefix --> attach to the left/right
        #
        if diffPos == (nodes[node] >> self.PREFIX_SHIFT):  # also true for head==0
            attachNode = newNode

        else:  # No node with the required prefix found --> create one
            attachNode = self._addPrefixNode(diffPos, key, newNode, node)

        self._attachToParent(key, parent, attachNode)
        return newNode

    def _addPrefixNode(self, prefix, key, newNode, prevNode):

        right = self._bitAt(key, prefix)

        node = self._addNode(prefix, right, newNode)
        self._setChildIndex(node, (1 - right), prevNode)
        return node

    def _attachToParent(self, key, parent, newNode):

        if parent:
            self._setChildIndex(parent, self._bitAt(key, self._nodes[parent] >> self.PREFIX_SHIFT),
                                newNode)  # where should the node be attached?
        else:
            self._head = newNode

    def _setChildIndex(self, pos, isRight, value):

        shift = isRight * self.INDEX_SHIFT
        mask = self.INDEX_MASK << shift

        self._nodes[pos] = (self._nodes[pos] & ~mask) + ((value << shift) & mask)

    def _iter(self, node: int):
        """
        Generator yields (nodeID, prefix, value) tuples.

        If prefix = 0: value is a key.
        If prefix > 0: value is a (leftNodeID, rightNodeID) tuple.
        """
        stack = deque()
        nodes = self._nodes
        nodeVal = nodes[node]

        if nodeVal:  # empty tree: key no. zero is empty
            while True:
                prefix = nodeVal >> self.PREFIX_SHIFT
                left = nodeVal & self.INDEX_MASK
                right = (nodeVal >> self.INDEX_SHIFT) & self.INDEX_MASK

                if not prefix:  # is leaf
                    if self.COMPACT_MODE:
                        left = (nodeVal >> 56) & 0x1
                        right = (nodeVal >> 57) & 0x1

                    if left:
                        yield node, prefix, self.key(self._id(nodeVal, 0))
                    if right:
                        yield node, prefix, self.key(self._id(nodeVal, 1))

                if prefix:  # prefix node
                    yield node, prefix, (left, right)

                    if left:
                        if right:
                            stack.append(right)
                        node = left
                        nodeVal = nodes[left]
                    elif right:
                        node = right
                        nodeVal = nodes[right]
                elif len(stack):
                    node = stack.pop()
                    nodeVal = nodes[node]
                else:
                    break
        return
