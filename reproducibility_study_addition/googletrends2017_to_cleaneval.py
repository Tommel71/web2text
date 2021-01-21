from bs4 import BeautifulSoup

path_in = "data/googletrends-2017/prepared_html/"
path_out = "data/googletrends-2017/cleaneval_style/"


import os
pages = os.listdir(path_in)

for path in pages:
    page_path = path_in + path

    with open(page_path, 'r', encoding="utf8") as reader:
        page = reader.read()

    soup = BeautifulSoup(page, "html.parser")
    spans = soup.find_all('span', __boilernet_label="1")

    outfile = path_out + path

    with open(outfile[:-4] + "txt" , 'w', encoding="utf8") as reader:
        for span in spans:
            a = "<p> " + ' '.join(span.text.split()) +  "\n\n"
            page = reader.write(a)



path_out_2 = "data/googletrends-2017/cleaneval_style2/"
for path in pages:
    page_path = path_in + path

    with open(page_path, 'r', encoding="utf8") as reader:
        page = reader.read()

    soup = BeautifulSoup(page, "html.parser")
    spans = soup.find_all('span', __boilernet_label="1")

    outfile = path_out_2 + path

    with open(outfile[:-4] + "txt" , 'w', encoding="utf8") as reader:
        for span in spans:
            a = ' '.join(span.text.split()) +  "\n"
            page = reader.write(a)

