

PATH='/data/virat-v2/videos_original'
OUTPUT_PATH='/data/virat-v2/video-info'

for video_file in "$PATH"/*
do
    basename=$(/usr/bin/basename $video_file)
    echo Processing $basename
    /usr/bin/ffprobe -v quiet ${video_file} -print_format json -show_format -show_streams > $OUTPUT_PATH/$basename.out
done

