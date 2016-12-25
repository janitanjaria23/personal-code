"""
Question URL:
http://www.geeksforgeeks.org/sum-elements-k1th-k2th-smallest-elements/
"""


def main():
    inp_list = [10, 2, 50, 12, 48, 13]
    k1 = 2
    k2 = 6

    sorted_inp_list = sorted(inp_list)

    print(sum(sorted_inp_list[k1:k2-1]))

if __name__ == '__main__':
    main()
