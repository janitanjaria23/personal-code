
def is_valid_int(n):
    try:
        return not(int(n) >> 32)
    except Exception as err:
        return False


def reverse_integer(n):
    # if is_valid_int(n):
    if n < 2 ** 31 or n > -2 ** 31:
        if n > 0:
            n_str = str(n)
            reversed_n_str = n_str[::-1]
        else:
            n_str = str(n)[1:]
            n_str = n_str
            reversed_n_str = '-' + n_str[::-1]
        # print n_str

        # print reversed_n_str, int(reversed_n_str)
    else:
        return 0
    if -(2 ** 31) < int(reversed_n_str) < (2 ** 31):
        return int(reversed_n_str)
    else:
        return 0


def test_new(A):
    A = str(A)
    if A[0] == '-':
        A = '-' + A[1:][::-1]
    else:
        A = A[::-1]
        A = int(A)
    if A > 2 ** 31 or A < -2 ** 31:
        return 0
    return A


def main():
    n = -1146467285
    # n = -1234567891
    # n = 12
    # n = -123
    # res = test_new(n)
    res = reverse_integer(n)
    print res


main()
