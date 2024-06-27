import os
import re
import requests


def get_soundgasm(url, save_path):
    r = requests.get(url)
    pattern = r'https://media.soundgasm.net.*?\.m4a'
    matches = re.findall(pattern, r.text)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cmd = "wget -P {} {}".format(save_path, matches[0])
    print(cmd)
    os.system(cmd)



def find_https_strings(text):
    pattern = r'https://\S+'
    matches = re.findall(pattern, text)
    return matches

            

if __name__ == "__main__":
    with open('/nas_dev/zio/text/2.txt') as f:
        for line in f:
            if find_https_strings(line.strip()):
                url = find_https_strings(line.strip())[0]
                spk = url.split('/')[-2]
                save_path = '/nas_dev/zio/listen_to_my_voice/{}'.format(spk)
                get_soundgasm(url, save_path)
