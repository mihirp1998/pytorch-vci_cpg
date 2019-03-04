if (( $# != 1 )); then
    echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
    exit
fi
hier=$1

modeldir=model

train="data/train"
eval="data/eval"
train_mv="data/train_mv"
eval_mv="data/eval_mv"

if [[ ${hier} == "0" ]]; then
  distance1=6
  distance2=6
  bits=16
  encoder_fuse_level=1
  decoder_fuse_level=1
elif [[ ${hier} == "1" ]]; then
  distance1=3
  distance2=3
  bits=16
  encoder_fuse_level=2
  decoder_fuse_level=3
elif [[ ${hier} == "2" ]]; then
  distance1=1
  distance2=2
  bits=8
  encoder_fuse_level=2
  decoder_fuse_level=1
  load_iter=36
  load_model_name=0228_fixbmvscale_L2_DL1_clip0.5_flip_--v-compress_--use-lstm--warp--fuse-encoder--use-bmv_0.0005_shrink2_batch32_patch64_dist1_2_bits8_ffi2
else
  echo "Usage: ./train.sh [0-2], e.g. ./train.sh 2"
  exit
fi

# Warning: with --save-out-img, output images are stored
# each time we run evaluation. This can take a lot of space
# when using a big evaluation dataset.
# (for the demo data it's okay.)


python -u train.py \
  --train ${train} \
  --eval ${eval} \
  --train-mv ${train_mv} \
  --eval-mv ${eval_mv} \
  --encoder-fuse-level ${encoder_fuse_level} \
  --decoder-fuse-level ${decoder_fuse_level} \
  --v-compress --warp --stack --fuse-encoder \
  --bits ${bits} \
  --distance1 ${distance1} --distance2 ${distance2} \
  --load-iter ${load_iter} \
  --load-model-name ${load_model_name} \
  --max-train-iters 10000 \
  --save-out-img
