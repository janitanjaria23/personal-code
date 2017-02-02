
def power_of_integer(n):
    if n == 0:
        return False
    if n == 1:
        return True
    else:
        for i in range(2, 2 ** 16):
            for j in range(2, 33):
                if i ** j == n:
                    return True
                if i ** j >= 2 ** 32:
                    break
    return False


def main():
    n = 7
    res = power_of_integer(n)
    print res


main()
