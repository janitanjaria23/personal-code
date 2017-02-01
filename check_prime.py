from math import sqrt


def check_prime(n):
    is_prime = True
    if n > 1:
        for i in range(2, int(sqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break
    else:
        is_prime = False
    print is_prime
    return is_prime


def check_prime_bug(n):
    upperLimit = int(n ** 0.5)
    if n > 1:
        for i in xrange(2, upperLimit + 1):
            if i < n and n % i == 0:
                return 0
        return 1
    else:
        return 0


def main():
    # n = 82944
    n = 4
    # check_prime(n)
    res = check_prime_bug(n)
    print res

main()