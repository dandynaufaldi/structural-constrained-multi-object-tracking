# https://stackoverflow.com/a/8991864
mat = [
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1],
]


def generate(index: int, array: list):
    global mat
    if len(array) == len(mat):
        yield array
    else:
        for i, val in enumerate(mat[index]):
            if val == 1:
                array.append(i)
                yield from generate(index + 1, array)
                array.pop()


def generate_stack(index: int, array: list):
    global mat
    stack = [(index, array)]
    while stack:
        index, array = stack.pop()
        print("panjang", len(stack))
        if len(array) == len(mat):
            yield array
        else:
            for i, val in enumerate(mat[index]):
                if val == 1:
                    arr: list = array.copy()
                    arr.append(i)
                    stack.append((index + 1, arr))


for perm in generate_stack(0, []):
    print(perm)
