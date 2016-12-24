import sys


def check_for_all_unique_characters(input_str):
    check_flag = 0
    for ch in input_str:
        value = ord(ch) - ord('a')
        if check_flag & (1 << value) > 0:
            return False
        check_flag |= value

    return True


def main():
    input_str = sys.argv[1]
    if check_for_all_unique_characters(input_str):
        return True
    else:
        return False


if __name__ == '__main__':
    flag = main()
    print(flag)
