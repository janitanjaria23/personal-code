def find_sqrt(n):
    start = 0
    end = n
    ans = 0
    while start <= end:
        mid = start + (end - start) / 2

        if (mid * mid) == n:
            return mid
        elif n > (mid * mid):
            start = mid + 1
            ans = mid
        else:
            end = mid - 1

    return ans


def main():
    n = 100
    res = find_sqrt(n)
    print res


main()
