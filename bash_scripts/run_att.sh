#!/bin/bash

dataset='coco' # flickr8k
id="att"
lr="5e-4"
while getopts 'I:' flag; do
  case "${flag}" in
    I) dataset="${OPTARG:-'coco'}" ;;
    d) decay="${OPTARG:-'15'}" ;;
    l) lr="${OPTARG:-'5e-4'}" ;;
    b) bs="${OPTARG:-'128'}" ;;
    *) error "Unexpected option ${flag}" ;;
  ':') echo "no value" ;;
  esac
done

if [ ${dataset?} = 'coco' ];then
  input_json=${input_json_coco?}
  input_label_h5=${input_label_h5_coco?}
  input_fc_dir=${input_fc_dir_coco?}
  input_att_dir=${input_att_dir_coco?}
  save_checkpoint_every="3000"
  val_images_use="5000"
  max_epochs="200"
  batch_size="128"
  learning_rate_decay_every=" --learning_rate_decay_every 15"
  more_args=""
  dataset=" --dataset coco"
elif [ ${dataset?} = 'conceptual' ];then # conceptual captions
  input_json=${input_json_conceptual?}
  input_label_h5=${input_label_h5_conceptual?}
  input_fc_dir=${input_fc_dir_conceptual?}
  input_att_dir=${input_att_dir_conceptual?}
  save_checkpoint_every="3000"
  val_images_use="10000"
  max_epochs="200"
  batch_size="128"
  learning_rate_decay_every=" --learning_rate_decay_every 15"
  more_args=""
  dataset=" --dataset conceptual"

elif [ ${dataset?} = 'flickr8k' ];then # conceptual captions
  input_json=${input_json_flickr8k?}
  input_label_h5=${input_label_h5_flickr8k?}
  input_fc_dir=${input_fc_dir_flickr8k?}
  input_att_dir=${input_att_dir_flickr8k?}
  more_args=" --input_encoding_size 128 --rnn_size 128 "
  save_checkpoint_every="3000"  # In iterations
  val_images_use="1000"
  max_epochs="100"  # In epoch
  batch_size="16"
  learning_rate_decay_every=" --learning_rate_decay_every 40"
  dataset=" --dataset flickr8k"

elif [ ${dataset?} = 'flickr30k' ];then # conceptual captions
  input_json=${input_json_flickr30k?}
  input_label_h5=${input_label_h5_flickr30k?}
  input_fc_dir=${input_fc_dir_flickr30k?}
  input_att_dir=${input_att_dir_flickr30k?}
  more_args=""
  save_checkpoint_every="1500"
  val_images_use="1014"
  max_epochs="100"
  batch_size=${bs?}
  learning_rate_decay_every=" --learning_rate_decay_every ${decay?}"
  dataset=" --dataset flickr30k"

else
cat <<- xx
  no such dataset [${dataset?}], please run 'coco', 'conceptual' or
  'flickr' dataset"
xx
fi


if [ -z ${jic_root_dir} ]; then
  ckpt_path="/cortex/users/gilad/DiscCaptioning_files/our_trained_models"
  ckpt_path+="/gumbel_models/pycharm_experiments/our/check_code/log_${id?}"
else
  ckpt_path="${jic_root_dir?}/pretrained_models/log_${id?}"
fi

mkdir -p ${ckpt_path?}
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "${ckpt_path?}
fi

cmd="python train.py --id $id --caption_model att2in2 --vse_model fc"
cmd+=" --share_embed 0 --input_json ${input_json?} --input_label_h5"
cmd+=" ${input_label_h5?} --input_fc_dir ${input_fc_dir?} --input_att_dir"
cmd+=" ${input_att_dir?} --batch_size ${batch_size?} --beam_size 1"
cmd+=" --learning_rate ${lr?}"
cmd+=" --learning_rate_decay_start 0 ${learning_rate_decay_every?}"
cmd+=" --scheduled_sampling_start 0 --checkpoint_path ${ckpt_path?}"
cmd+=" ${start_from?}"
cmd+=" --save_checkpoint_every ${save_checkpoint_every?} --language_eval 1"
cmd+=" --val_images_use ${val_images_use?}"
cmd+=" --max_epochs ${max_epochs?} --vse_loss_weight 0"
cmd+=" --retrieval_reward_weight 0"
cmd+=" --initialize_retrieval"
cmd+=" ${jic_root_dir?}/pretrained_models/log_fc_con/model_vse-best.pth"
cmd+=" --phase 2 ${dataset?}"
cmd+=" ${more_args?}"

echo "Running the command [${cmd?}]"
${cmd?}


