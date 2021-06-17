import requests
import bs4
import re
import argparse
from io import StringIO
from html.parser import HTMLParser

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def process_text(text, http_filter=True, filter_text=True):
    if filter_text:
        text = re.sub('"', "'", text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r"<[^>]+>|&nbsp;", '', text)
        text = strip_tags(text)
    if http_filter:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = '. '.join([i for i in text.split('.')])

    # review = re.sub(r'\w*\d\w*', ' ', review)  # Удаление цифр
    text = re.sub(r"\s+", ' ', text)  # Удаление лишних пробелов # Проболы, табуляция, переносы
    return text


def parser_rss(url):
    items = ['title', 'description', 'pubDate', 'link', 'full-text']  # full-text
    txt = []

    full_page = requests.get(f'{url}', headers=headers)
    soup = bs4.BeautifulSoup(full_page.content, 'xml')  # 'html.parser'
    data = soup.select('item')

    for i in range(len(data)):
        text = []
        for item in items:
            try:
                text1 = process_text(data[i].select(item)[0].text, http_filter=False)
            except:
                text1 = 'None'
            text.append(text1)
        txt.append(text)
    return txt


def parser_news(url, fl, add_title=True):
    n = ''
    d = ''
    title = ''
    full_page = requests.get(f'{url}', headers=headers)
    soup = bs4.BeautifulSoup(full_page.content, 'html.parser')
    if fl == 0:
        get = soup.find_all('div', {'class': 'full-news-content'})
        if add_title:
            title = get[0].select('h1')[0].text
        news = get[0].find('div', {'class': 'full-news-text'}).select('p')
    elif fl == 1:
        get = soup.find_all('div', {'class': 'ln-content-holder'})
        if add_title:
            title = get[1].select('h2')[0].text
        news = get[1].find('div', {'class': 'article-text'}).select('p')
    else:
        get = soup.find_all('article')
        if add_title:
            title = get[0].select('h1')[0].text
        d = get[0].select('h4')[0].text
        news = get[0].find('div', {'class': 'news_detail_text'}).select('p')

    for new in news:  # [:-2]
        n += new.text + ' '
    a, b = process_text(title), process_text(f"{d} {n}")
    return a, re.sub(a, "", b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rss_input", default=None, type=str, required=True,
                        help="Path to rss list file")
    parser.add_argument("--path_out", default=None, type=str, required=True,
                        help="Path to output rss news")
    parser.add_argument("--add_title", action="store_true")
    args = parser.parse_args()

    rss_input = args.rss_input
    Path_out = args.path_out
    urls = []
    Names = []
    data = []

    with open(rss_input, 'r', encoding='UTF-8') as f1:
        for i in f1:
            a = i.split()
            n = ''
            if len(a) > 2:
                for j in a[:-1]:
                    n += j + ' '
                urls.append(a[-1])
                Names.append(n)

    for url, name in zip(urls, Names):
        print(name)
        try:
            data.append(parser_rss(url))
        except:
            print(f'Error rss: {name}')
            data.append([])
    with open(f'{Path_out}/rss_links.txt', 'w', encoding='UTF-8') as f1:
        with open(f'{Path_out}/rss.txt', 'w', encoding='UTF-8') as f:
            if args.add_title:
                for name, d in zip(Names, data):
                    n = name.split()
                    for i in d:
                        if n[0] == '+':
                            print(process_text(f'{i[0]} {i[-1]} =>', filter_text=False), file=f)
                            print(i[-2], file=f1)
                        elif n[0] == '+(descr)':
                            print(process_text(f'{i[0]} {i[1]} =>', filter_text=False), file=f)
                            print(i[-2], file=f1)
                        elif n[0] == '-':
                            try:
                                if n[1] == 'Сибирь':
                                    parser_data = parser_news(i[-2], 0)
                                elif n[1] == 'ГИБДД':
                                    parser_data = parser_news(i[-2], 1)
                                elif n[1] == 'Портал':
                                    parser_data = parser_news(i[-2], 2)
                                print(f'{parser_data[0]} {parser_data[1]}=>', file=f)
                                print(i[-2], file=f1)
                            except:
                                print('Error parser_data')

            else:
                for name, d in zip(Names, data):
                    n = name.split()
                    for i in d:
                        if n[0] == '+':
                            print(process_text(f'{i[-1]} =>', filter_text=False), file=f)
                            print(i[-2], file=f1)
                        elif n[0] == '+(descr)':
                            print(process_text(f'{i[1]} =>', filter_text=False), file=f)
                            print(i[-2], file=f1)
                        elif n[0] == '-':
                            try:
                                if n[1] == 'Сибирь':
                                    parser_data = parser_news(i[-2], 0, False)
                                elif n[1] == 'ГИБДД':
                                    parser_data = parser_news(i[-2], 1, False)
                                elif n[1] == 'Портал':
                                    parser_data = parser_news(i[-2], 2, False)
                                print(f'{parser_data[1]}=>', file=f)
                                print(i[-2], file=f1)
                            except:
                                print('Error parser_data')


if __name__ == "__main__":
    main()
