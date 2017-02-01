
def merge_intervals(intervals, new_interval):
    ind = None
    new_ind = None
    st = None
    end = None
    result_list = []
    first_elements = [elem[0] for elem in (intervals + [new_interval])]

    first_elements.sort()

    for i in range(0, len(first_elements)):
        if first_elements[i] == new_interval[0]:
            ind = i
            break
    if ind:
        if intervals[ind - 1][0] < new_interval[0] < intervals[ind - 1][1]:
            st = intervals[ind - 1][0]

            if new_interval[1] < intervals[ind - 1][1]:
                end = intervals[ind - 1][1]
            else:
                new_ind = ind - 1
                for i in range(ind, len(intervals)):
                    if intervals[i][0] < new_interval[0] < intervals[i][1]:
                        if new_interval[1] > intervals[i][1]:
                            new_ind = i
                            continue
                        else:
                            new_ind = i
                            continue
                    else:
                        end = new_interval[1]

                    # if intervals[i][0] < new_interval[1] < intervals[i][1]:
                    #     continue
                    # else:
                    #     new_ind = i
                    #     break
                if not end:
                    end = intervals[new_ind][1]

    for i in range(0, len(intervals)):
        if st and end:
            if i < ind - 1 or i >= ind:
                result_list.append(intervals[i])
            else:
                result_list.append([st, end])
                i = new_ind + 1
    print result_list


def custom_comp(a, b):
    if a[0] < b[0]:
        return -1
    elif a[0] > b[0]:
        return 1
    else:
        return 0


def merge_intervals_new(intervals, new_interval):
    if new_interval[0] > new_interval[1]:
        temp_val = new_interval[0]
        new_interval[0] = new_interval[1]
        new_interval[1] = temp_val
    full_list = intervals + [new_interval]

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
    # intervals = [[1, 3], [6, 9]]
    # intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    intervals = [[1, 2], [3, 6]]
    # intervals = [[3, 5], [8, 10]]
    # new_interval = [2, 5]
    # new_interval = [4, 9]
    new_interval = [7, 7]
    # new_interval = [10, 3]

    result_list = merge_intervals_new(intervals, new_interval)
    print result_list


main()
