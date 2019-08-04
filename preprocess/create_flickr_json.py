
import json
import sys
sys.path.append("coco-caption")

flickr8_path = '/cortex/users/gilad/DiscCaptioning_files/dataset_flickr8k.json'
flickr30_path = '/cortex/users/gilad/DiscCaptioning_files/dataset_flickr30k' \
                '.json'
coco_path = '/cortex/users/gilad/DiscCaptioning_files/dataset_coco.json'

# captions_val2014_path = 'coco-caption/annotations/captions_val2014.json'

captions_8k_path = '/cortex/users/gilad/DiscCaptioning_files' \
                   '/captions_flickr8k.json'

paths = [(flickr8_path, 'flickr8'), (flickr30_path, 'flickr30'), (coco_path,
                                                                  'coco')]
with open(flickr8_path, 'rb') as f:
    flickr8_json = json.load(f)

with open(flickr30_path, 'rb') as f:
    flickr30_json = json.load(f)

with open(coco_path, 'rb') as f:
    coco_json = json.load(f)

# with open(captions_val2014_path, 'rb') as f:
#     captions_val2014 = json.load(f)

with open(captions_8k_path, 'rb') as f:
    captions_8k = json.load(f)

# for key in captions_val2014['info']:
#     print(captions_val2014['info'][key])

