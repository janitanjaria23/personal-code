

def sort_by_color(l):
    white = []
    red = []
    blue = []
    for elem in l:
        if elem == 0:
            red.append(elem)
        elif elem == 1:
            white.append(elem)
        else:
            blue.append(elem)

    print red + white + blue
    return red + white + blue


def main():
    l = [0, 2, 0, 2]
    res = sort_by_color(l)
    print res


main()