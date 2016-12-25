"""
Question URL:
http://www.geeksforgeeks.org/sort-an-array-according-to-absolute-difference-with-given-value/
"""

import collections


def main():
    inp_list = [2, 6, 8, 3]  # sys.argv[1]
    base_num = 5  # sys.argv[2]
    data_dict = collections.defaultdict(list)
    for elem in inp_list:
        data_dict[abs(elem - base_num)].append(elem)

    inp_list = []
    for diff, elems in data_dict.items():
        inp_list += elems

    print(inp_list)


if __name__ == '__main__':
    main()
