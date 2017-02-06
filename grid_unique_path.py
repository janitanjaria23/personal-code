import math

def factorial(n):
    if n == 1:
        return n
    else:
        return n * factorial(n - 1)


def count_unique_paths(a, b):
    total = (a - 1) + (b - 1)
    print factorial(total)
    print factorial(a - 1), factorial(b - 1)
    cnt = math.factorial(total) / (math.factorial(a - 1) * math.factorial(b-1))
    print cnt
    return cnt


def main():
    a = 3
    b = 3
    res = count_unique_paths(a, b)
    print res


main()
