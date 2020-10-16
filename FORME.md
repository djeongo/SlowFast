source ~/venv/bin/activate
# Train
python tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50_multigrid.yaml DATA.PATH_TO_DATA_DIR /home/pius/sdc1/data/ DATA.PATH_PREFIX /home/pius/sdc1/data/Charades_v1_rgb NUM_GPUS 1 TRAIN.BATCH_SIZE 16 

# Test
python tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50_multigrid.yaml DATA.PATH_TO_DATA_DIR /home/pius/sdc1/data/ DATA.PATH_PREFIX /home/pius/sdc1/data/Charades_v1_rgb NUM_GPUS 1 TEST.CHECKPOINT_FILE_PATH /home/pius/cs230/SlowFast.git/SLOWFAST_16x8_R50_multigrid.pkl TRAIN.ENABLE False


https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md