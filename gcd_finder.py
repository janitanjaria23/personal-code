
def find_gcd(a, b):
    if b != 0:
        rem = a % b
        while rem != 0:
            a = b
            b = rem
            rem = a % b
            print a, b, rem
        # print b
        return b
    else:
        return a


def main():
    a = 4
    b = 6
    gcd_res = find_gcd(a, b)
    print gcd_res


main()
