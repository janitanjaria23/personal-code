import sys


def main():
    base_str = sys.argv[1]
    compare_to_str = sys.argv[2]
    if base_str and compare_to_str:
        if ''.join(sorted(base_str)) == ''.join(sorted(compare_to_str)):
            return True

    return False


if __name__ == '__main__':
    flag = main()
    print(flag)
