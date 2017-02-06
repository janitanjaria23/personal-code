def rearrange_array(l):
    for i in range(0, len(l)):
        print l[i], l[l[i]]
        l[i] += l[l[i]] % len(l) * len(l)
    print l

    for i in range(0, len(l)):
        l[i] /= len(l)

    print l


def main():
    l = [0, 4, 1, 2, 3, 5]
    # l = [1, 0]
    rearrange_array(l)


main()
