import os
import re
import requests_html

def get_literotica(url, save_path):
    session = requests_html.HTMLSession() 
    r = session.get(url)
    pattern = r'https://uploads.literotica.com/audio.*?\.m4a'
    matches = re.findall(pattern, r.text)
    cmd = "wget -P {} {}".format(save_path, matches[0])
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    get_literotica('https://www.literotica.com/s/jess-cums-8-another-gusher', './')