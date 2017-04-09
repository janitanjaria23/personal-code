def minimum(a, b):
    if a < b:
        return a
    else:
        return b


def min_characters(input_str, l, h):
    if l > h:
        return float('Inf')
    if l == h:
        return 0
    if l == h - 1:
        if input_str[l] == input_str[h]:
            return 0
        else:
            return 1
    if input_str[l] == input_str[h]:
        return min_characters(input_str, l + 1, h - 1)
    else:
        return min(min_characters(input_str, l, h - 1), min_characters(input_str, l + 1, h)) + 1


def compute_lps_array(input_str):
    m = len(input_str)
    lps = [0] * m
    length = 0
    lps[0] = 0
    i = 1
    while i < m:
        if input_str[i] == input_str[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps


def get_min_characters_to_be_added(input_str):
    reversed_str = ''.join(reversed(input_str))
    concat_str = input_str + "$" + reversed_str

    lps = compute_lps_array(concat_str)

    return len(input_str) - lps[-1]


def main():
    # input_str = "geeks"
    input_str = "AACECAAAA"
    input_str = "hqghumeaylnlfdxfi"
    input_str = "banana"

    # res = min_characters(input_str, 0, len(input_str) - 1)
    res = get_min_characters_to_be_added(input_str)
    print res

main()
