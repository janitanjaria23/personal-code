from collections import Counter


def single_number(l):
    # info_dict = {}
    c = Counter(l)
    for k, v in c.iteritems():
        if v <= 1:
            return k

    return -1


def single_number_new(l):
    s = 0
    for elem in l:
        s ^= elem

    print s
    return s


def main():
    l = [1, 2, 2, 3, 1]
    # res = single_number(l)
    res = single_number_new(l)
    print res


main()
