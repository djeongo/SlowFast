# 
INPUT_DIR=$1
BASENAME=$(basename $INPUT_DIR)
OUTPUT_DIR=~/sdc1/data/VIRAT-resized/$BASENAME


mkdir -p $OUTPUT_DIR
echo $INPUT_DIR $OUTPUT_DIR

find $INPUT_DIR -iname '*.JPG' -exec convert \{} -verbose -set filename:basename "%[basename]" -resize 320x240\> "$OUTPUT_DIR/%[filename:basename].JPG" \;