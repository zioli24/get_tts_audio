from pytube import YouTube
from pytube import Playlist
from tqdm import tqdm


#download single youtube mp4 file or a whole playlist, need sign in first
def get_youtube(url, save_path):
    print("downloading video from {}".format(url)) 
    if url.find("playlist") == -1: 
        yt = YouTube(url = url, use_oauth = True, allow_oauth_cache = True)
        yt.streams.filter(only_audio=True).first().download(save_path)    

    else:
        p = Playlist(url)
        for video_url in tqdm(p.video_urls):     
            yt = YouTube(url = video_url, use_oauth = True, allow_oauth_cache = True)
            yt.streams.filter(only_audio=True).first().download(save_path)
    print("successful download from {}".format(url)) 


if __name__ == "__main__":
    with open('/nas_dev/zio/text/4.txt') as f:
        for line in f:
            url = line.strip()
            save_path = '/nas_dev/zio/listen_to_my_voice'
            get_youtube(url, save_path)