def get_column_title(column_num):
    column_title = ""
    while column_num > 0:
        q = column_num / 26
        temp_rem = (column_num % 26 - 1)
        if temp_rem >= 0:
            rem = 65 + (column_num % 26 - 1)
            column_num /= 26
        else:
            rem = 65 + 25  # 'Z'
            column_num = column_num / 26 - 1
        column_title += chr(rem)

        # print column_num

    # print column_title[::-1]
    return column_title[::-1]


def main():
    # column_num = 53
    column_num = 943566
    # test_new(column_num)
    column_title = get_column_title(column_num)
    print column_title


main()
