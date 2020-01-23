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


for perm in generate(0, []):
    print(perm)
