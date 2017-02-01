
def custom_comp(a, b):
    if a[0] < b[0]:
        return -1
    elif a[0] > b[0]:
        return 1
    else:
        return 0


def merge_intervals_new(intervals):
    full_list = intervals

    # first_elements = [elem[0] for elem in full_list]

    full_list.sort(cmp=custom_comp)

    temp_list = [full_list[0]]

    for i in range(1, len(full_list)):
        if temp_list[-1][0] <= full_list[i][0] <= temp_list[-1][1]:
            if full_list[i][1] > temp_list[-1][1]:
                temp_list[-1][1] = full_list[i][1]
        else:
            temp_list.append(full_list[i])

    print full_list
    print temp_list
    return temp_list


def main():
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    result_list = merge_intervals_new(intervals)
    print result_list


main()

