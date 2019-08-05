#!/bin/bash

### IMPORTANT ###
# run the bash script from terminal as follow:
# nohup sh our_scripts/run_joint.sh (list of arguments) &> /dev/null &
# see Example 6

# new flags
#    learning rate
#    learning rate decay
#    batch size
: ' Command line running examples
All flags have default values, send to scripts only unique flags for this run
Example 1 - Optimization gumbel, gpu 3, lr 3e-5 and
disc weight 0.05:
bash_scripts/run_joint.sh -G 3 -o gumbel -l 3e-5 -D 0.05

Example 2 - Optimization multinomial_soft, gpu 2, lr_decay 0.7 and
learning rate decay every 12:
bash_scripts/run_joint.sh -G 2 -o multinomial_soft -d 0.7 -e 12

Example 3 - Optimization reinforce alternate, gpu 1, batch size 50,
caption loss weight 0.5 and vse loss wight 0.1:
bash_scripts/run_joint.sh -G 1 -o reinforce -b 50 -c 0.5 -v 0.1

Example 4 - For Optimization reinforce only speaker training, gpu 0,
caption loss weight 0.5:
bash_scripts/run_joint.sh -G 0 -o reinforce_speaker -c 0.5

Example 5 - Optimization reinforce alternate, run on CPU
caption loss weight 0.1, vse loss 0.1:
bash_scripts/run_joint.sh -C 1 -o reinforce -c 0.12345 -v 0.1

'

# Default values
gpu=0
discriminative=0.01
temperature=1
lr=5e-4
decay=0.8
every=15
batch=128
caption=0
vse=0
optimization='gumbel'
phase=""
reinforceBaselineType='gt'
sharedEmbbeding=0
cpu=0
prob=0.25
prob_gumbel_multinomial=""
softmax_cooling=0
max_epochs=350
softmax_cooling_decay_factor=""
use_docker=0
dataset='coco'  # flickr30
annealing=0 # temperature_annealing_factor
temperature_annealing_factor=""
annealing_every=0
num_iteration_for_annealing=""

while getopts 'S:G:D:t:l:d:e:b:v:c:o:r:E:C:p:O:u:I:a:n:' flag; do
  case "${flag}" in
    G) gpu="${OPTARG:-0}" ;;
    D) discriminative="${OPTARG:-0.01}" ;;
    t) temperature="${OPTARG:-5}" ;;
    l) lr="${OPTARG:-5e-4}" ;;
    d) decay="${OPTARG:-0.8}" ;;
    e) every="${OPTARG:-15}" ;;
    b) batch="${OPTARG:-128}" ;;
    v) vse="${OPTARG:-0}" ;;
    c) caption="${OPTARG:-0}" ;;
    o) optimization="${OPTARG:-'gumbel'}" ;;
    r) reinforceBaselineType="${OPTARG:-'gt'}" ;;
    E) sharedEmbbeding="${OPTARG:-0}" ;;
    C) cpu="${OPTARG:-0}" ;;
    p) prob="${OPTARG:-0.25}" ;;
    O) softmax_cooling="${OPTARG:-0}" ;;
    u) use_docker="${OPTARG:-0}" ;;
    I) dataset="${OPTARG:-'coco'}" ;;
    a) annealing="${OPTARG:-0.93}" ;;
    n) annealing_every="${OPTARG:-500}" ;;

    *) error "Unexpected option ${flag}" ;;
  ':') echo "no value" ;;
  esac
done


# cider = 1 - discriminative
cider=$(echo 1-${discriminative?} | bc)
conda_env="conda activate discaption"
source_env="source activate discaption"
fixed_dir="_C${cider?}_LR${lr?}_D${decay?}_E${every?}"

# Add softmax_cooling if softmax_cooling bigger than 0
if (( $(echo "${softmax_cooling?} > 0" |bc -l) )); then
    fixed_dir+="_O${softmax_cooling?}"
fi

fixed_dir+="_BS${batch?}"    ## include the other flags
if [ ${use_docker?} -eq 1 ]; then
  jic_root_dir="/project/artifacts/DC/our_trained_models/models/"
fi

# If jic_root_dir does not exists, create it and all its subdirectories
mkdir -p ${jic_root_dir?}
mkdir -p ${jic_root_dir?}/gumbel
mkdir -p ${jic_root_dir?}/gumbel_softmax
mkdir -p ${jic_root_dir?}/gumbel_speaker
mkdir -p ${jic_root_dir?}/multinomial
mkdir -p ${jic_root_dir?}/multinomial_soft
mkdir -p ${jic_root_dir?}/multinomial_speaker
mkdir -p ${jic_root_dir?}/reinforce
mkdir -p ${jic_root_dir?}/reinforce_speaker
mkdir -p ${jic_root_dir?}/reinforce_listener
mkdir -p ${jic_root_dir?}/json_dir

if [ ${use_docker?} -eq 1 ]; then
    jic_code_dir='/project/DC/'
	USER="docker"
	if [ -z "$NGC_JOB_ID" ]; then
	    docker_data_dir='/data'
	else
	    # Inside a NGC docker, we first copy the data to local machine and
	    # extract it
	    docker_data_dir='/tmp'
	    cp /data/*.tar /tmp
	    mkdir -p /tmp/cocobu_att /tmp/cocotalk_fc
	    apt install pigz
	    pigz -dc /tmp/cocobu_att.tar | tar xf - -C /tmp/cocobu_att
	    pigz -dc /tmp/cocotalk_fc.tar | tar xf - -C /tmp/cocotalk_fc
	    rm /tmp/*.tar 
	fi
fi

# set dirs and flags according to chosen optimization
if [ ${optimization?} = 'gumbel' ];then
  echo 'gumbel optimization was chosen'
  dir="G${discriminative?}_T${temperature?}_TA${annealing?}_AE"
  dir+="${annealing_every?}${fixed_dir?}"
  alternating_turn=" --alternating_turn speaker "
  alternating_turn+="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  smoothing_temperature="--gumbel_temp ${temperature?}"
  retrieval_reward="--retrieval_reward gumbel"
  reinforceBaselineType=""
  temperature_annealing_factor=" --gumbel_temperature_annealing_factor"
  temperature_annealing_factor+=" ${annealing?}"
  num_iteration_for_annealing=" --num_iteration_for_annealing"
  num_iteration_for_annealing+=" ${annealing_every?} "
elif [ ${optimization?} = 'gumbel_softmax' ];then
  echo 'gumbel_softmax optimization was chosen'
  dir="GS${discriminative?}_T${temperature?}_P${prob?}${fixed_dir?}"
  alternating_turn=" --alternating_turn speaker "
  alternating_turn+="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  smoothing_temperature="--gumbel_temp ${temperature?}"
  prob_gumbel_multinomial="--prob_gumbel_softmax ${prob?}"
  softmax_cooling_decay_factor="--softmax_cooling_decay_factor "
  softmax_cooling_decay_factor+="${softmax_cooling?}"
  retrieval_reward="--retrieval_reward gumbel_softmax"
  reinforceBaselineType=""
elif [ ${optimization?} = 'multinomial' ];then
  echo 'multinomial optimization was chosen'
  dir="M${discriminative?}_T${temperature?}${fixed_dir?}"
  alternating_turn="--alternating_turn speaker "
  alternating_turn+="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  smoothing_temperature="--multinomial_temp ${temperature?}"
  retrieval_reward="--retrieval_reward multinomial"
  reinforceBaselineType=""
elif [ ${optimization?} = 'multinomial_soft' ];then
  echo 'multinomial_soft optimization was chosen'
  dir="MS${discriminative?}_T${temperature?}_P${prob?}${fixed_dir?}"
  alternating_turn="--alternating_turn speaker "
  alternating_turn+="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  smoothing_temperature="--multinomial_temp ${temperature?}"
  prob_gumbel_multinomial="--prob_multinomial_soft ${prob?}"
  softmax_cooling_decay_factor="--softmax_cooling_decay_factor "
  softmax_cooling_decay_factor+="${softmax_cooling?}"
  retrieval_reward="--retrieval_reward multinomial_soft"
  reinforceBaselineType=""
elif [ ${optimization?} = 'reinforce' ];then
  echo 'reinforce alternating optimization was chosen'
  dir="R${discriminative?}_CAP${caption?}_V${vse?}_BSL_"
  dir+="${reinforceBaselineType?}${fixed_dir?}"
  alternating_turn="--alternating_turn speaker "
  alternating_turn+="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  retrieval_reward="--retrieval_reward reinforce"
  reinforceBaselineType=" --reinforce_baseline_type gt"
  smoothing_temperature=""
# For ablation study
elif [ ${optimization?} = 'reinforce_listener' ];then
  echo 'reinforce listener optimization was chosen'
  dir="rl${discriminative?}${fixed_dir?}"
  #  Train only without listener
  alternating_turn="--alternating_turn listener"
  is_alternating=" --is_alternating 1"
  retrieval_reward="--retrieval_reward reinforce"
  reinforceBaselineType=" --reinforce_baseline_type gt"
  smoothing_temperature=""
  max_epochs=500
  vse=1
  cider=0
  caption=0
elif [ ${optimization?} = 'reinforce_speaker' ];then
  echo 'reinforce only speaker training optimization was chosen'
  dir="r${discriminative?}${fixed_dir?}"
  alternating_turn=""
  is_alternating=" --is_alternating 0"
  smoothing_temperature=""
  phase="--phase 3"
  retrieval_reward="--retrieval_reward reinforce"
  reinforceBaselineType=""
elif [ ${optimization?} = 'gumbel_speaker' ];then
  echo 'gumbel only speaker training optimization was chosen'
  dir="g${discriminative?}_T${temperature?}${fixed_dir?}"
  alternating_turn=""
  is_alternating=" --is_alternating 0"
  smoothing_temperature="--gumbel_temp ${temperature?}"
  phase="--phase 3"
  retrieval_reward="--retrieval_reward gumbel"
  reinforceBaselineType=""
elif [ ${optimization?} = 'multinomial_speaker' ];then
  echo 'multinomial only speaker training optimization was chosen'
  dir="m${discriminative?}_T${temperature?}${fixed_dir?}"
  alternating_turn=""
  is_alternating=" --is_alternating 0"
  smoothing_temperature="--multinomial_temp ${temperature?}"
  phase="--phase 3"
  retrieval_reward="--retrieval_reward multinomial"
  reinforceBaselineType=""
else
   echo "Unknown optimization"

fi
ckpt_path="${jic_root_dir?}/${optimization?}/${dir?}"
# set dataset files
if [ ${dataset?} = 'coco' ];then
  input_json=${input_json_coco?}
  input_label_h5=${input_label_h5_coco?}
  input_fc_dir=${input_fc_dir_coco?}
  input_att_dir=${input_att_dir_coco?}
  save_checkpoint_every="3000"
  val_images_use=" --val_images_use 5000"
  max_epochs="350"
  more_args=""
elif [ ${dataset?} = 'flickr30k' ];then # flickr30k captions
  input_json=${input_json_flickr30k?}
  input_label_h5=${input_label_h5_flickr30k?}
  input_fc_dir=${input_fc_dir_flickr30k?}
  input_att_dir=${input_att_dir_flickr30k?}
  save_checkpoint_every="3000"  # In iterations
  val_images_use=" --val_images_use 1014"
  max_epochs="300"  # In epoch
  more_args=""
fi

####print to user####
cat <<- xx
##################################
	#retrieval_reward_weight (discriminative) is ${discriminative?}
	#temperature (for gumbel\multinomial) is ${temperature?}
	#cider weight (cider) is ${cider?}
	#${retrieval_reward?}
	#target dir is ${ckpt_path?}
	#Chosen dataset is ${dataset?}
	##################################
xx
####finish print to user####


if [ ! -d ${ckpt_path?} ]; then
  copy_model_cmd="bash bash_scripts/copy_model.sh att att_d${discriminative?}"
  copy_model_cmd+=" ${ckpt_path?} ${jic_root_dir?} ${jic_code_dir?}"
  ${copy_model_cmd?}
fi

cuda_cmd="CUDA_VISIBLE_DEVICES=${gpu?}"
cmd=" python train.py"
cmd+=" --id att_d${discriminative?}"
cmd+=" --caption_model att2in2"
cmd+=" --vse_model fc --share_embed ${sharedEmbbeding?}"
if [ ${use_docker?} -eq 1 ]; then
    cmd+=" --input_json /project/artifacts/DC/cocotalk.json"
    cmd+=" --input_label_h5 /project/artifacts/DC/cocotalk_label.h5"
    cmd+=" --input_fc_dir ${docker_data_dir}/cocotalk_fc"
    cmd+=" --input_att_dir ${docker_data_dir}/cocobu_att"
else
    cmd+=" --input_json ${input_json?}"
    cmd+=" --input_label_h5 ${input_label_h5?}"
    cmd+=" --input_fc_dir ${input_fc_dir?}"
    cmd+=" --input_att_dir ${input_att_dir?}"
fi

cmd+=" --batch_size ${batch?} --seq_per_img 1 --beam_size 1"
cmd+=" --learning_rate ${lr?}"
cmd+=" --learning_rate_decay_rate ${decay?} --learning_rate_decay_start 0"
cmd+=" --learning_rate_decay_every ${every?} --scheduled_sampling_start 0"
cmd+=" --checkpoint_path ${ckpt_path?} --start_from ${ckpt_path?}"
cmd+=" --save_checkpoint_every ${save_checkpoint_every?}"
cmd+=" --language_eval 1 ${val_images_use?} --max_epochs ${max_epochs?}"
cmd+=" ${retrieval_reward?} --retrieval_reward_weight ${discriminative?}"
cmd+=" --vse_loss_weight ${vse?}"
cmd+=" --initialize_retrieval ${jic_root_dir?}/pretrained_models/"
cmd+="/log_fc_con/model_vse-best.pth"
cmd+=" --cider_optimization ${cider?}"
cmd+=" --caption_loss_weight ${caption?} --rank_eval 1"
cmd+=" ${smoothing_temperature?} ${is_alternating?}"
cmd+=" ${alternating_turn?} --rank_on_gen_captions"
cmd+=" --listener_stage_1_model_path ${jic_root_dir?}/pretrained_models"
cmd+="/log_fc_con/model_vse-best.pth"
cmd+=" --speaker_stage_2_optimizer_path ${jic_root_dir?}/pretrained_models"
cmd+="/log_att/optimizer.pth"
cmd+=" --speaker_stage_2_model_path"
cmd+=" ${jic_root_dir?}/pretrained_models/log_att/model.pth"
cmd+=" ${phase?} ${reinforceBaselineType?}"
cmd+=" ${prob_gumbel_multinomial?}"
cmd+=" ${softmax_cooling_decay_factor?} --dataset ${dataset?}"
cmd+=" ${more_args?} ${temperature_annealing_factor?}"
cmd+=" ${num_iteration_for_annealing?}"

cd ${jic_code_dir?}
if [ ! -d "logs/" ]; then
  mkdir "logs/"
fi

if [ ${use_docker?} -eq 1 ]; then
    # If we are inside a docker
    echo ${cmd?}
    ${cmd?}

else
# If we use GPU add the number of gpu to python command
  if [ ${cpu?} -eq 0 ] ; then
#    cmd="${cuda_cmd?}${cmd?}"
     export ${cuda_cmd?}
  fi
  echo ${cmd?}
  ${cmd?}
fi


