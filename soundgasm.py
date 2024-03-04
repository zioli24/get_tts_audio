import os
import re
import requests


def get_soundgasm(url, save_path):
    r = requests.get(url)
    pattern = r'https://media.soundgasm.net.*?\.m4a'
    matches = re.findall(pattern, r.text)
    cmd = "wget -P {} {}".format(save_path, matches[0])
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    get_soundgasm('https://soundgasm.net/u/VanillaLust_/Baby-Sitting-and-Dick-Sitting-the-DILF-Neighbor', './')