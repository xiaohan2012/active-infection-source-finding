from bisect import bisect_left


def insert(seq, keys, item, keyfunc):
    """Insert an item into the sorted list using separate corresponding
    keys list and a keyfunc to extract key from each item.
    """
    k = keyfunc(item)  # get key
    i = bisect_left(keys, k)  # determine where to insert item
    keys.insert(i, k)  # insert key of item in keys list
    seq.insert(i, item)


def remove(seq, keys, item, keyfunc):
    """remove a item
    """
    k = keyfunc(item)  # get key
    i = bisect_left(keys, k)  # determine where to remove item
    keys.pop(i)  # remove key of item in keys list
    seq.pop(i)
