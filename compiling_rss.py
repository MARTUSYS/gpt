import datetime
import argparse
import os


def KMP(t, s):
    x = -1
    v = [0] * len(s)
    for i in range(1, len(s)):
        k = v[i - 1]
        while k > 0 and s[k] != s[i]:
            k = v[k - 1]
        if s[k] == s[i]:
            k = k + 1
        v[i] = k
    k = 0
    for i in range(len(t)):
        while k > 0 and s[k] != t[i]:
            k = v[k - 1]
        if s[k] == t[i]:
            k = k + 1
        if k == len(s):
            x = i - len(s) + 1 + 1
            break
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rss_input", default=None, type=str, required=True,
                        help="Path to rss list file")
    parser.add_argument("--path_out", default=None, type=str, required=True,
                        help="Path to output rss news")
    parser.add_argument("--len_data", default=5, type=int)
    args = parser.parse_args()

    rss_input = args.rss_input
    Path_out = args.path_out
    len_data = args.len_data + 3

    names = []
    for root, dirs, files in os.walk(rss_input):
        for f in files:
            if f[-3:] == 'rss':
                names.append(f'{root}/{f}')

    data = []
    for name in names:
        with open(name, 'r', encoding='UTF-8') as f:
            data.append(f.readlines())

    for i in range(len(names)):
        a = names[i].split("/")[-1][:-4]
        if KMP(a, 'title') != -1:
            names[i] = f'<container type="title" model="{a}">\n'
        elif KMP(a, 'text') != -1:
            names[i] = f'<container type="text" model="{a}">\n'
        else:
            names[i] = f'<container type="description" model="{a}">\n'

    with open(f'{rss_input}/rss_links.txt', 'r', encoding='UTF-8') as f:
        links = f.readlines()

    date_now = datetime.datetime.now().date()

    with open(f'{Path_out}/compiled_rss_{date_now}.xml', 'w', encoding='UTF-8') as f:
        f.write(
            '<rss xmlns:yandex="http://news.yandex.ru" xmlns:media="http://search.yahoo.com/mrss/" version="2.0">\n'
            '<channel>\n'
            '<title>FEFU GPT-3</title>\n'
            '<description>Title and description</description>\n'
            '<link>...</link>\n'
        )

        for l_d in range(0, len(data[0]), len_data):
            f.write('<item>\n')
            f.write(f'<link>{links[(l_d + 1) // len_data][:-1]}</link>\n')
            f.write(f'<yandex:full-text>{data[0][l_d][:-4]}</yandex:full-text>\n')
            for d in range(len(data)):
                f.write(names[d])  # container
                for l_d_item in range(l_d + 2, l_d + len_data - 1):  # +2 компенсация title, -1 компенсация ----
                    f.write(f'<p>{data[d][l_d_item][:-1]}</p>\n')
                f.write(f'</container>\n')
            f.write('</item>\n')

        f.write(
            '</channel>\n'
            '</rss>'
        )


if __name__ == "__main__":
    main()
