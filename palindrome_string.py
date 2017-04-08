
def check_for_palindrome_string(input_str):
    processed_str = ""
    for ch in input_str.lower():
        if ch.isalpha() or ch.isdigit():
            processed_str += ch

    # print processed_str
    i = 0
    j = len(processed_str) - 1
    palindrome_flag = 1
    while i <= j:
        if processed_str[i] != processed_str[j]:
            palindrome_flag = 0
            break
        i += 1
        j -= 1

    return palindrome_flag


def main():
    # input_str = "A man, a plan, a canal: Panama"
    input_str = "race a car"
    res = check_for_palindrome_string(input_str)
    print res

main()