


def get_all_paren(n_open,n_close,tmp_s,all_paren = []):
    if n_open == n_close and n_open == 0:
        all_paren.append(tmp_s)
        return all_paren
    elif n_open == n_close:
        return get_all_paren(n_open-1,n_close,tmp_s+'(',all_paren )
    elif n_open == 0:
        tmp_s += n_close * ')'
        all_paren.append(tmp_s)
        return all_paren
    elif n_open < n_close:
        get_all_paren(n_open,n_close-1,tmp_s+')',all_paren)
        get_all_paren(n_open -1, n_close ,tmp_s+'(',all_paren)
        return all_paren



def all_parens(n):
    n_open,n_close = n,n
    return get_all_paren(n_open,n_close,'')


def main():
    print(all_parens(3))


if __name__ == '__main__':
    main()