"""
Question URL:
http://www.geeksforgeeks.org/minimize-sum-product-two-arrays-permutations-allowed/
"""


def main():
    first_list = [6, 1, 9, 5, 4]
    second_list = [3, 4, 8, 2, 4]
    sum_of_product = 0
    sorted_first_list = sorted(first_list)
    sorted_second_list = sorted(second_list, reverse=True)

    for elem in zip(sorted_first_list, sorted_second_list):
        sum_of_product += (elem[0]*elem[1])

    print(sum_of_product)

if __name__ == '__main__':
    main()