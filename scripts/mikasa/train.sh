D=/data/nrf3/usrdata/eric/dataset/mikasa-robo-vla-rlds
DIRS=$(for t in shell_game_push_vla_v0 intercept_medium_vla_v0 remember_color_5_vla_v0 take_it_back_vla_v0 remember_shape_and_color_3x3_vla_v0; do printf "%s," "$D/$t/1.0.0"; done)
CKPT=/home/eric/usrdata/llm_weights/TurboVLA/mikasa_baseline_upscale
mkdir -p $CKPT
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port 29501 experiments/mikasa/train.py \
  --dataset_dirs "${DIRS%,}" \
  --stats_path experiments/mikasa/configs/mikasa5_stats.json \
  --text_layout_path '/home/eric/TurboVLA/experiments/mikasa/configs/online_text_layout.json' \
  --text_padding_length 44 \
  --stats_key mikasa5 \
  --resize_images_to 256 \
  --chunk_size 12 \
  --dinov3_path /home/eric/TurboVLA/pretrained/dinov3 \
  --bert_path google-bert/bert-base-uncased \
  --allow_hf_download \
  --pretrained_init_ckpt /home/eric/TurboVLA/pretrained/groundingdino/groundingdino_swint_ogc.pth \
  --batch_size 64 --grad_accum_steps 1 \
  --head_lr 1e-4 --dinov3_lr 1e-4 --min_lr_ratio 0.1 \
  --max_steps 40000 --warmup_steps 4000 --save_steps 4000 \
  --shuffle_buffer 1250 \
  --checkpoint_dir $CKPT \
  2>&1 | tee -a $CKPT/train.log
