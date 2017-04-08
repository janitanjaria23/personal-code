
def reverse_string(input_str):
    word_stack = []
    result_list = []
    word_str = ""

    for ch in input_str.strip():
        if ch != " " and ch != "":
            word_str += ch
        else:
            if word_str:
                word_stack.append(word_str)
            word_str = ""
    if word_str:
        word_stack.append(word_str)

    # print word_stack
    while len(word_stack) > 0:
        result_list.append(word_stack.pop())

    # print result_list

    return ' '.join(result_list)


def main():
    # input_str = "janit anjaria"
    input_str = "the sky     is blue"
    res = reverse_string(input_str)
    print res

main()
