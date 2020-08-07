SOURCE=/home/viraj-uk/Documents/tensorflow/research/
CODE_HOME=/home/viraj-uk/Documents/ssd_aoeiii/
PYTHONPATH=$PYTHONPATH:$SOURCE/slim:$SOURCE
export PYTHONPATH

#export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim

MODEL_DIR=$CODE_HOME/models/model/
mkdir $MODEL_DIR

PIPELINE_CONFIG_PATH=$CODE_HOME/models/pipeline.config

cd $SOURCE
#
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
