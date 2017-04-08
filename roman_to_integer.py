
def convert_roman_to_integer(roman_str):
    roman_info_dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result_val = 0
    prev_val = 0
    for ch in roman_str:
        current_val = roman_info_dict[ch]
        if current_val <= prev_val:
            result_val += roman_info_dict[ch]
        else:
            result_val -= prev_val
            result_val += (current_val - prev_val)
        prev_val = current_val
        # print result_val

    return result_val


def main():
    roman_str = "LXXIV"
    res = convert_roman_to_integer(roman_str)
    print res

main()
