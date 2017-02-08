def custom_power(x, n):
    # temp = -1
    if n == 0:
        return 1
    temp = custom_power(x, n / 2)
    if n % 2 == 0:
        return temp * temp
    else:
        return x * temp * temp


def compute_pow(x, n, d):
    l = range(0, d + 1)
    start = 0
    end = len(l) - 1
    res = -1
    while start <= end:
        mid = start + (end - start) / 2

        if custom_power(x, n) % d == l[mid]:
            res = mid
            break
        elif custom_power(x, n) % d > l[mid]:
            start = mid + 1
        else:
            end = mid - 1

    print res
    return res


def bitwise_pow(x, n, d):
    a = x % d
    t = 1
    if x == 0 and n == 0 and d == 1:
        return (x ** n) % d

    while n > 0:
        if n & 1:
            t = (t * a) % d

        n >>= 1
        a = (a * a) % d

    print t
    return t


def main():
    x = 2
    # x = 71045970
    n = 3
    # n = 41535484
    d = 3
    # d = 64735492
    res = bitwise_pow(x, n, d)
    # res = compute_pow(x, n, d)
    print res


main()
