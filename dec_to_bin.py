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
    return ''.join([str(elem) for elem in digits_list])


def main():
    # n = 6
    n = 11
    base = 2  # base to convert to
    res = dec_to_bin(n, base)
    print res


main()
