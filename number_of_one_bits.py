def dec_to_bin(n, base):
    digits_list = []
    if n > 0:
        while n > 0:
            rem = n % base
            digits_list.append(rem)
            n /= base

        digits_list.reverse()

    else:
        digits_list = [0]

    # print digits_list
    return digits_list, ''.join([str(elem) for elem in digits_list])


def count_one_bits(n):
    digits_list, bin_rep = dec_to_bin(n, 2)
    cnt = 0
    for elem in bin_rep:
        if elem == '1':
            cnt += 1

    return cnt


def count_one_bits_new(n):

    cnt = 0

    while n > 0:
        n &= n - 1
        cnt += 1

    print cnt
    return cnt


def main():
    n = 11
    # res = count_one_bits(n)
    res = count_one_bits_new(n)
    print res


main()
