def _recursive_merge(dict1, dict2):
    """
    Recursively merge two dictionaries. Combines sub-dictionaries where both have the same key.
    """
    result = dict1.copy()  # Start with a shallow copy of the first dictionary
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _recursive_merge(result[key], value)
        else:
            result[key] = value
    return result


def rec_gen_trie(l):
    if l == []:
        return dict()
    else:
        head = l[0]
        tail = l[1:]

        if head == []:
            return rec_gen_trie(tail)
        else:
            h_head = head[0]
            h_tail = head[1:]
            rec = rec_gen_trie(tail)
            return _recursive_merge({h_head:rec_gen_trie([h_tail])} ,rec)
        

def gen_trie(columns):
    return rec_gen_trie(list(map(lambda x:(x.split("_")),columns)))