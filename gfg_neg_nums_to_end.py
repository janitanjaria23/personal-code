"""
Question URL:
http://www.geeksforgeeks.org/move-ve-elements-end-order-extra-space-allowed/
"""

import sys


def main():
    inp_list = [1, 2, -3, -1, -2]

    for idx in range(1, len(inp_list)):
        val = inp_list[idx]

        if val > 0:
            continue
        idy = idx - 1
        while idy >= 0 and inp_list[idy] > 0:
            inp_list[idy + 1] = inp_list[idy]
            idy -= 1
        inp_list[idy + 1] = val

    print(inp_list)


if __name__ == '__main__':
    main()
