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
    get_youtube('https://www.youtube.com/watch?v=vN7DFZazSik&ab_channel=UFC', './')
    #get_youtube('https://www.youtube.com/playlist?list=PLFKG1h4xM85TFcvzMG7elWVaV-JUMs21R', './')