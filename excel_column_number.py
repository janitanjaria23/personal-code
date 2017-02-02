
def get_column_number(input_str):
    reversed_str = input_str[::-1]
    sum = 0
    for i in range(0, len(reversed_str)):
        val = (ord(reversed_str[i]) % 65) + 1
        # print (26 ** i) * val
        sum += (26 ** i) * val

    # print sum
    return sum


def main():
    input_str = "BA"
    column_number = get_column_number(input_str)
    print column_number


main()
