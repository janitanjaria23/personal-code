from math import sqrt


def generate_all_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    prime_numbers_list = []
    for i in range(2, int(sqrt(n)) + 1):
        if is_prime[i]:
            j = 2
            while i * j <= n:
                is_prime[i * j] = False
                j += 1

    for i in range(2, len(is_prime)):
        if is_prime[i]:
            prime_numbers_list.append(i)

    # print prime_numbers_list
    return prime_numbers_list


def main():
    n = 3
    res = generate_all_primes(n)
    print res


main()
