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


def matrix_search(m, x):
    ll = [row[-1] for row in m]  # finding last elements of each row
    ind = -1
    for elem in ll:
        if x <= elem:
            ind = ll.index(elem)
            break
        else:
            continue

    if ind == -1:
        return 0
    else:
        result_index = binary_search(m[ind], x)
        print ind, result_index
        if result_index != -1:
            return 1
        else:
            return 0
        # return result_index


def main():
    m = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]
    x = 21
    res = matrix_search(m, x)
    print res


main()
