
def square_sum(n):
    ans = []
    a = 1
    if n > 1:
        while a * a <= n:
            b = 1
            while b * b <= n:
                if a * a + b * b == n:
                    newEntry = [a, b]
                    rev_newEntry = [b, a]
                    if rev_newEntry not in ans:
                        ans.append(newEntry)
                b += 1
            a += 1
        print ans
    return ans


def main():
    n = 1
    res = square_sum(n)
    print res

main()
