Results obtained with this code are provided in the paper:
Joint optimization for cooperative image captioning, 
G. Vered, G. Oren, Y. Atzmon and G. Chechik, International Conference on Computer Vision (ICCV), 2019 

##  Installation

Download the code 
```
git clone https://github.com/vgilad/JointImageCaptioning.git
```
Set the environmnt
```
conda create -n discaption python=3.6
source activate discaption
yes | conda install numpy=1.14.3
yes | conda install pytorch=0.4.1 cuda92 -c pytorch
yes | conda install scipy=1.1.0
yes | conda install pandas=0.23.0
yes | conda install h5py=2.8.0
yes | conda install six=1.11.0
yes | conda install tensorflow=1.8.0
yes | conda install torchvision=0.2.0
yes | conda install scikit-image=0.13.1
yes | conda install matplotlib=2.2.2
yes | conda install -c conda-forge python-lmdb
```
Add to your .bashrc the location of the downloaded preprocessed data.
Set the 'jic_root_dir' for desirable output location and the 'jic_code_dir' for the location of project directory.
```
export input_json_coco="/file/location/cocotalk.json"  
export input_label_h5_coco="/file/location/cocotalk_label.h5"  
export input_fc_dir_coco="/directory/location/cocotalk_fc"  
export input_att_dir_coco="/directory/location/cocobu_att"  
  
export jic_code_dir="location/of/code/"  
export jic_root_dir="location/for/your/future/trained/models"  
```

## Download pre-processed features or compute them 

#### Option 1: Download pre-processed data

From https://github.com/ruotianluo/DiscCaptioning use the following [link](https://drive.google.com/drive/folders/1Z9bfvkRT5YyikmNgzPbybezYj9mi4TE2)
download the data files to the right locations set above in your .bashrc with the commands below
```
mkdir -p ${input_fc_dir_coco}
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=10ZLGwTQyd7EztJI7Hm_DILIhv01V9ine "`dirname ${input_fc_dir_coco}`/cocotalk_fc.tar" && tar -C ${input_fc_dir_coco} -xvf "`dirname ${input_fc_dir_coco}`/cocotalk_fc.tar" 
mkdir -p ${input_att_dir_coco}
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=17MjWOU6rOTqJjdB201yCzKJOkCUXTUqc "`dirname ${input_att_dir_coco}`/cocobu_att.tar" && tar -C ${input_att_dir_coco} -xvf "`dirname ${input_att_dir_coco}`/cocobu_att.tar"
mkdir -p `dirname $input_json_coco`
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=1r-ANklnok4zy7lVO6a7pJKvxwc-G-jej ${input_json_coco}
mkdir -p `dirname $input_label_h5_coco`
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=1W8SR0xbKxSLdXJP94ULxgtZmzd_kMr3B ${input_label_h5_coco}

```
#### Option 2: Preprocess the features
Follow the instructions [here](https://github.com/ruotianluo/DiscCaptioning)

## Train the model
### Pretraining

#### Option 1: Download pre-trained models
Pretrained models can be downloaded from [link](https://drive.google.com/drive/folders/1Iw-9KmHJYbQ3-jRLLCsNVs9-i5I_vsAi?usp=sharing), or with the following command:  
```
mkdir -p ${jic_root_dir}/pretrained_models  
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=1rHWUKMDEw9YdVxTFQmAysKy9eI454c9f ${jic_root_dir}/pretrained_models/log_att.tar && tar -C ${jic_root_dir}/pretrained_models -xvf ${jic_root_dir}/pretrained_models/log_att.tar
${jic_code_dir}/preprocess/gdown.pl https://drive.google.com/open?id=1ALvSH-NpaRBB2n-qTzauGdm0knfWjzly ${jic_root_dir}/pretrained_models/log_fc_con.tar && tar -C ${jic_root_dir}/pretrained_models -xvf ${jic_root_dir}/pretrained_models/log_fc_con.tar
```
#### Option 2: Pre-trained he models yourself
Pretrain the discriminator with human captions.
```
bash_scripts/run_fc_con.sh
```
Pretrain the generator with MLE loss
Locate log_fc_con directory under jic_root_dir/pretrained_models/
Run:
```
bash_scripts/run_att.sh
```

### 2. Discriminative phase
Results will be saved under {jic_root_dir}/chosen_optimization.
##### If you pretrained the speaker and listener, locate their output directories under ${jic_root_dir}/pretrained_models/
Run:

```
bash_scripts/run_joint.sh
```
#### Running Examples
Example run with GPU, gumbel optimization:
```
bash_scripts/run_joint.sh -S 19 -G 1 -o gumbel -D 0.01 -d 0.75 -l 5e-3 -t 8
```
Example run with GPU, Reinforce optimization:
```
bash_scripts/run_joint.sh -S 02 -G 1 -o reinforce -D 0.8 -d 0.8 -l 5e-3 -v 0.1 -r gt
```
Example run with CPU, gumbel optimization:
```
bash_scripts/run_joint.sh -S 03 -C 1 -o gumbel -D 0.0005 -d 0.75 -l 5e-3 -t 1
```
See more examples in bash_scripts/run_joint.sh


## Citation and Acknowledgments
```
@InProceedings{vered2019cooperative,
   author = {Vered, Gilad and Oren, Gal and Atzmon, Yuval and Chechik Gal},
   title = {Joint optimization for cooperative image captioning},
   booktitle = {International Conference on Computer Vision (ICCV)},
   year = {2019}
}
```
The code for this project is based on https://github.com/ruotianluo/DiscCaptioning, we thank the authors for making teir cide available and for their support of the project.

This research was supported by an Israel science foundation grant 737/18
