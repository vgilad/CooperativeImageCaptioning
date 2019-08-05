cd #!/bin/sh

if [ -f /.dockerenv ]; then
    load_from="/project/artifacts/DC/our_trained_models/models/"
else
#    load_from="/cortex/users/gilad/DiscCaptioning_files/our_trained_models/models/"
  load_from="${4?}/" # root_dir
fi
load_from+="pretrained_models/log_att/"
cp -r ${load_from?} $3
cd $3
mv infos_$1-best.pkl infos_$2-best.pkl 
mv infos_$1.pkl infos_$2.pkl 
if [ -f /.dockerenv ]; then 
    cd /project/DC
else
#    cd /home/lab/vgilad/PycharmProjects/JointImageCaptioning
    cd ${5?} # working_dir
fi
