# youtube-dl

`youtube-dl --proxy socks5://127.0.0.1:1080/ --write-auto-sub -f bestvideo+bestaudio` + [URL]

只看视频和音频的格式：-F：
下载相应版本的视频：-f 可以跟上面的形式，也可以跟137+140

ffmpeg应该自动合成audio和video：由于YouTube的1080p及以上的分辨率都是音视频分离的,所以我们需要分别下载视频和音频,可以使用137+140这样的组合. 如果系统中安装了ffmpeg的话, youtube-dl 会自动合并下下好的视频和音频, 然后自动删除单独的音视频文件

# ffmpeg

将ffmpeg加入到PATH： 
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/，
https://blog.csdn.net/Chanssl/article/details/83050959
