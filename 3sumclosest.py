import math


def find_3sumclosest(l, target):
    min_val = float('Inf')
    result = 0
    l.sort()
    for i in range(0, len(l)):
        j = i + 1
        k = len(l) - 1
        while j < k:
            sum_val = l[i] + l[j] + l[k]
            diff = math.fabs(sum_val - target)
            if diff == 0:
                return sum_val

            if diff < min_val:
                min_val = diff
                result = sum_val

            if sum_val <= target:
                j += 1
            else:
                k -= 1
    return result


def main():
    input_list = [-1, 2, 1, -4]
    target_val = 1
    res = find_3sumclosest(input_list, target_val)
    print res


main()
