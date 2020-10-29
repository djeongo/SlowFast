# IN_DATA_DIR="/home/pius/Downloads/VIRAT/videos_original"

# VIRAT-V2
# IN_DATA_DIR="/data/actev-data-repo/corpora/VIRAT-V2"
# OUT_DATA_DIR="/data/virat-v2"

# VIRAT-V1
IN_DATA_DIR="/data/actev-data-repo/corpora/VIRAT-V1/combined"
OUT_DATA_DIR="/data/virat-v1/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

ls $IN_DATA_DIR

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do

  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  # ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
