def heap_push(ls, val):
    index = len(ls)
    ls.append(val)
    parent = int((index - 1) / 2)
    while ls[index] < ls[parent]:
        ls[index], ls[parent] = ls[parent], ls[index]
        index = parent
        parent = int((index - 1) / 2)


def heap_pop(ls):
    if len(ls) == 1: return ls.pop()
    item = ls[0]
    ls[0] = ls.pop()
    index = -1
    next_i = 0
    while next_i != index and int(next_i * 2 + 1) < len(ls):
        index = next_i
        next_i = shift(ls, index)
    return item


def heapify(ls):
    if len(ls) < 2: return
    curr = int((len(ls) - 2) / 2)
    while curr >= 0:
        shift(ls, curr)
        curr -= 1


def shift(ls, node):
    left = node * 2 + 1
    right = node * 2 + 2
    has_right = right < len(ls)
    min_index = left
    if has_right and ls[right] < ls[left]: min_index = right
    if ls[min_index] < ls[node]:
        ls[min_index], ls[node] = ls[node], ls[min_index]
        node = min_index
    return node
