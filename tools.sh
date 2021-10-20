# python3 utils/select_samples.py --prob_path /mnt/nas/share-map/experiment/liuye/Mobili/ImageSeg \
#     --image_path /mnt/nas/share-map/experiment/liuye/Mobili/Image \
#     --output_path /mnt/nas/share-map/experiment/sijin/03dataset/DMSegV2/raw \
#     --selected_num 500

# python3 data_mining.py --image_folder /mnt/nas/share-map/experiment/sijin/03dataset/Curvelanes/Curvelanes/train \
#     --output_folder /mnt/nas/share-map/experiment/sijin/03dataset/Curvelanes/Curvelanes/train_mined_data_for_seg

python3 data_mining.py --image_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/SSLAD-2D/labeled/train \
    --output_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/train_mined_data --image_num 2000

python3 data_mining.py --image_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/SSLAD-2D/labeled/val \
    --output_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/val_mined_data --image_num 2000

python3 data_mining.py --image_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/SSLAD-2D/labeled/test \
    --output_folder /mnt/nas/share-map/experiment/sijin/03dataset/SODA2D/test_mined_data --image_num 2000
