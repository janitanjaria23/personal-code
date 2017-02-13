import numpy as np


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


def count(a, b):
    first_index = count_occurence(a, b, True)
    if first_index == -1:
        print 0
        return 0
    else:
        last_index = count_occurence(a, b, False)
        print last_index - first_index + 1
        return last_index - first_index + 1


def find_intersection(a, b):
    intersection_list_index = []
    intersection_dict = {}
    intersection_list = []
    for i in range(0, len(b)):
        search_result = binary_search(a, b[i])
        # ind = b.index(elem)

        if search_result != -1 and i not in intersection_dict and search_result not in intersection_dict.values():
            # intersection_list_index.append((i, search_result))
            intersection_dict[i] = search_result

    for ind, val in intersection_dict.iteritems():
        intersection_list.append(b[ind])

    print intersection_list
    return intersection_list


def find_intersection_new(a, b):
    intersection_list = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
            if a[i] < b[j]:
                i += 1
            elif b[j] < a[i]:
                j += 1
            else:
                intersection_list.append(b[j])
                # print b[j]
                i += 1
                j += 1

    print intersection_list
    return intersection_list


def main():
    # a = [1, 2, 3, 3, 4, 5, 6]
    a = [1, 3, 8, 10, 13, 13, 16, 16, 16, 18, 21, 23, 24, 31, 31, 31, 33, 35, 35, 37, 37, 38, 40, 41, 43, 47, 47, 48, 48, 52, 52, 53, 53, 55, 56, 60, 60, 61, 61, 63, 63, 64, 66, 67, 67, 68, 69, 71, 80, 80, 80, 80, 80, 80, 81, 85, 87, 87, 88, 89, 90, 94, 95, 97, 98, 98, 100, 101 ]
    # b = [3, 3, 5]
    b = [5, 7, 14, 14, 25, 28, 28, 34, 35, 38, 38, 39, 46, 53, 65, 67, 69, 70, 78, 82, 94, 94, 98]
    # res = find_intersection(a, b)
    res = find_intersection_new(a, b)
    print res


main()
