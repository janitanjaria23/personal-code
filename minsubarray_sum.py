def find_min_subarray_sum(l, target_sum):
    result = len(l)
    start = 0
    sum = 0
    i = 0
    exists = False

    if len(l) == 1 or not l:
        return 0

    while i < len(l):
        if sum >= target_sum:
            exists = True

            if start == (i - 1):
                return 1

            result = min(result, i - start)
            sum -= l[start]
            start += 1
        else:
            if i == len(l):
                break
            sum += l[i]
            i += 1
    if exists:
        return result
    else:
        return 0


def main():
    l = [2, 3, 1, 2, 4, 3]
    s = 7
    res = find_min_subarray_sum(l, s)
    print res


main()
