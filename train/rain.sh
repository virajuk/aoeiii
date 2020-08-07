#CODE_HOME=/home/viraj-uk/Documents/tensorflow/models/research
#CODE_HOME=/home/viraj-uk/Documents/ssd_aoeiii/
#PYTHONPATH=PYTHONPATH::$CODE_HOME:$CODE_HOME/slim

cd /home/viraj-uk/Documents/tensorflow/models/research/
#export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

MODEL_DIR=/home/viraj-uk/Documents/ssd_aoeiii/models/model/exp001
#mkdir $MODEL_DIR

PIPELINE_CONFIG_PATH=/home/viraj-uk/Documents/ssd_aoeiii/models/pipeline.config
#PIPELINE_CONFIG_PATH=$CODE_HOME/models/pipeline.config
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

#cd /home/viraj-uk/Documents/tensorflow/models/research/

#import sys
#echo python --version

python -c 'import sys; print sys.version_info'

python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
