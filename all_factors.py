from math import sqrt


def all_factors(n):
    factor_list = []
    second_factor_list = []
    if n > 1:
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0:
                factor_list.append(i)
                if i != sqrt(n):
                    second_factor_list.append((n/i))

        for i in range(len(second_factor_list)-1, -1, -1):
            factor_list += [second_factor_list[i]]
    else:
        factor_list.append(1)
    print factor_list
    return factor_list


def main():
    # n = 12
    # n = 1
    # n = 36
    n = 82944
    all_factors(n)


main()
