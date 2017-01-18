import string
from math import floor


def convert_to_base62(inp, b=62):
    """
    converts an input string to base62
    :param inp: input text/string
    :param b: base
    :return: resultant base62 encoded string/text
    """
    if b <= 0 or b > 62:
        return 0

    base = string.digits + string.lowercase + string.uppercase
    r = inp % b
    res = base[r]
    q = floor(inp / b)
    while q:
        r = q % b
        q = floor(q / b)
        res = base[int(r)] + res

    return res


def convert_to_base10(num, b=62):
    """
    Converts the given string/text to base10.
    :param num: input string/integer in b62
    :param b:
    :return:
    """
    base = string.digits + string.lowercase + string.uppercase
    limit = len(num)
    res = 0
    for i in xrange(limit):
        res = b * res + base.find(num[i])
    return res