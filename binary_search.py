def binary_search(l, element):
    n = len(l)
    start = 0
    end = n - 1
    while start <= end:
        mid = start + (end - start) / 2
        if l[mid] == element:
            return mid
        elif element < l[mid]:
            end = mid - 1
        else:
            start = mid + 1
    return -1


def main():
    l = [2, 3, 5, 13, 16, 71]
    element = 14
    res = binary_search(l, element)
    print res


main()
