def max_continuous(l, target_val=1):
    result_list = []
    max_cnt = 0
    print len(l)

    if len(l) == 0:
        return result_list

    else:
        for i in range(0, len(l)):
            cnt_left, cnt_right = 0, 0
            if l[i] == 0:

                j = i - 1
                while l[j] == target_val and j >= 0:
                    cnt_left += 1
                    j -= 1
                j = i + 1

                while l[j] == target_val and j < (len(l) - 1):

                    cnt_right += 1
                    j += 1
                    # print i, j, len(l) - 1

                if (cnt_left + cnt_right + 1) > max_cnt:
                    max_cnt = cnt_left + cnt_right + 1
                    result_list = range((i - cnt_left), (i + cnt_right) + 1)
                    # print result_list
    return result_list


def max_count(l, m):
    wl, wr, bestl, best_window = 0, 0, 0, 0
    zero_count = 0

    while wr < len(l):
        if zero_count <= m:
            if l[wr] == 0:
                zero_count += 1
            wr += 1
        if zero_count > m:
            if l[wl] == 0:
                zero_count -= 1
            wl += 1

        if wr - wl > best_window:
            best_window = wr - wl
            bestl = wl
    print bestl, best_window
    result_list = []
    for i in range(bestl, bestl + best_window):
        result_list.append(i)
    # result_list = range(bestl, best_window)
    return result_list


def main():
    l = [1, 1, 0, 1, 1, 0, 0, 1, 1]
    # l = [ 0, 1, 1, 1 ]
    # target_val = 1
    m = 1
    # m = 0
    # res = max_continuous(l, target_val)
    res = max_count(l, m=m)
    print res


main()
