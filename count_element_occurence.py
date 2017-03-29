
def count_occurence(a, b, first_search):
    n = len(a)
    start = 0
    end = n - 1
    cnt = 0
    flag = -1
    while start <= end:
        mid = start + (end - start) / 2
        if a[mid] == b:
            flag = mid
            if first_search:
                end = mid - 1
            else:
                start = mid + 1
            # cnt += 1
        elif b < a[mid]:
            end = mid - 1
        else:
            start = mid + 1

    return flag


def main():
    a = [5, 7, 7, 8, 8, 8, 10]
    b = 7
    first_index = count_occurence(a, b, True)
    if first_index == -1:
        print 0
        return 0
    else:
        last_index = count_occurence(a, b, False)
        print last_index - first_index + 1
        return last_index - first_index + 1

main()
