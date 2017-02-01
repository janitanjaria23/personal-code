
def max_subarray(l):
    max_sum = -float('INF') - 1
    cum_sum = 0
    st = 0
    end = 0
    #     temp_l = []
    temp_end = 0
    temp_st = 0
    temp_sum = 0
    for i in range(0, len(l)):
        cum_sum += l[i]
        if cum_sum >= max_sum and cum_sum >= 0:
            max_sum = cum_sum
            end = i
        if cum_sum < 0 or l[i] < 0:
            if end - st + 1 >= temp_end - temp_st + 1 :
                temp_end = end
                temp_st = st
            cum_sum = 0
            st = i + 1

            # print max_sum, cum_sum, st, end

    if max_sum >= 0:
        if len(l[temp_st: temp_end + 1]) > len(l[st: end + 1]) and sum(l[temp_st: temp_end + 1]) == sum(l[st: end + 1]):
            return l[temp_st: temp_end + 1]

        elif len(l[temp_st: temp_end + 1]) == len(l[st: end + 1]) and sum(l[st: end + 1]) == sum(l[temp_st: temp_end + 1]):
            if temp_st < st:
                return l[temp_st: temp_end + 1]
            else:

                return l[st: end + 1]
        else:
            return l[st: end + 1]
    else:
        return []


def main():
    l = [1, 2, 5, -7, 2, 3]
    # l = [0, 0, -1, 0]
    # l = [-1, -1, -1, -1]
    # l = [1967513926, 1540383426, -1303455736, -521595368]
    # l = [-846930886, -1714636915, 424238335, -1649760492]
    # l = [756898537, -1973594324, -2038664370, -184803526, 1424268980]
    res = max_subarray(l)
    print res


if __name__ == '__main__':
    main()