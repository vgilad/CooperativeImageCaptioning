#!/bin/bash

dataset='coco'

while getopts 'I:' flag; do
  case "${flag}" in
    I) dataset="${OPTARG:-'coco'}" ;;
    *) error "Unexpected option ${flag}" ;;
  ':') echo "no value" ;;
  esac
done

id="fc_con"
if [ -z ${jic_root_dir} ]; then
  #ckpt_path="/cortex/users/gilad/DiscCaptioning_files/our_trained_models
  # /models/"
  ckpt_path="/cortex/users/gilad/DiscCaptioning_files/our_trained_models"
  ckpt_path+="/debug_models/log_${id?}"
else
  ckpt_path="${jic_root_dir?}/pretrained_models/log_${id?}"
fi
mkdir -p ${ckpt_path?}
if [ ! -f ${ckpt_path?}"/infos_"${id?}".pkl" ]; then
start_from=""
else
start_from="--start_from "${ckpt_path?}
fi

# set dataset files
if [ ${dataset?} == 'coco' ];then
  input_json=${input_json_coco?}
  input_label_h5=${input_label_h5_coco?}
  input_fc_dir=${input_fc_dir_coco?}
  input_att_dir=${input_att_dir_coco?}
  val_images_use="5000"
  save_checkpoint_every="3000"
  batch_size=128
  more_args=""
elif [ ${dataset?} == 'conceptual' ];then # conceptual captions
  input_json=${input_json_conceptual?}
  input_label_h5=${input_label_h5_conceptual?}
  input_fc_dir=${input_fc_dir_conceptual?}
  input_att_dir=${input_att_dir_conceptual?}
  val_images_use="10000"
  save_checkpoint_every="75000"
  batch_size=128
  more_args=""
elif [ ${dataset?} == 'flickr30k' ];then # conceptual captions
  input_json=${input_json_flickr30k?}
  input_label_h5=${input_label_h5_flickr30k?}
  input_fc_dir=${input_fc_dir_flickr30k?}
  input_att_dir=${input_att_dir_flickr30k?}
  val_images_use="1014"
  save_checkpoint_every="1000"
  batch_size=128
#  more_args=" --input_encoding_size 128 --vse_embed_size 256"
  more_args=""
else
cat <<- xx
  no such dataset [${dataset?}], please run 'coco', 'conceptual' or
  'flickr' dataset
xx
fi

cmd="python train.py"
cmd+=" --id ${id?} --caption_model fc --vse_model fc"
cmd+=" --share_embed 0"
cmd+=" --input_json ${input_json?}"
cmd+=" --input_label_h5 ${input_label_h5?}"
cmd+=" --input_fc_dir ${input_fc_dir?}"
cmd+=" --input_att_dir ${input_att_dir?}"
cmd+=" --batch_size ${batch_size?} --beam_size 1"
cmd+=" --learning_rate 5e-4 --learning_rate_decay_start 0"
cmd+=" --learning_rate_decay_every 15"
cmd+=" --scheduled_sampling_start 0"
cmd+=" --checkpoint_path ${ckpt_path?} ${start_from?}"
cmd+=" --save_checkpoint_every ${save_checkpoint_every?}"
cmd+=" --language_eval 0 --max_epochs 30"
cmd+=" --val_images_use ${val_images_use?}"
cmd+=" --vse_loss_weight 1 --caption_loss_weight 0"
cmd+=" --rank_eval 1 --phase 1"
cmd+=" --dataset ${dataset?} ${more_args?}"

echo "Running the command [${cmd?}]"

${cmd?}

