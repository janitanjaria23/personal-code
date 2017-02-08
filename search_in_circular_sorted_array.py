
def search_elem(l, x):
    n = len(l)
    start = 0
    end = n - 1

    while start <= end:
        mid = start + (end - start ) /2

        if l[mid] == x:
            return mid

        if l[mid] <= l[end]:
            if l[mid] < x <= l[end]:
                start = mid + 1
            else:
                end = mid - 1
        else:
            if l[start] <= x < l[mid]:
                end = mid - 1
            else:
                start = mid + 1
    return -1


def main():
    l = [4, 5, 6, 7, 0, 1, 2]
    x = 0
    res = search_elem(l, x)
    print res


main()
