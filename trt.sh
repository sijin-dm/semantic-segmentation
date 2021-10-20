python3 -m runx.runx scripts/pytorch2trt.yml -i
# python3 utils/infer_trt_model.py --save --model_name ocrnet_trt.NaiveLiteHRNet_trt.pth --img_folder /mnt/nas/share-map/common/DMCar/rawdata/20210705024852/images_00021
# python3 utils/infer_trt_model.py --save --model_name ocrnet_trt.HRNetW18_trt.pth --img_folder /mnt/nas/share-map/common/DMCar/rawdata/20210705024852/images_00021
# python3 utils/infer_trt_model.py --save --model_name ocrnet_trt.HRNet_trt.pth --img_folder /mnt/nas/share-map/common/DMCar/rawdata/20210705024852/images_00021
# python3 utils/infer_trt_model.py --save --model_name ocrnet_trt.HRNet_Mscale_trt.pth --img_folder /mnt/nas/share-map/common/DMCar/rawdata/20210705024852/images_00021
# python3 utils/infer_trt_model.py --save --model_name ocrnet_trt.DDRNet23_Slim_trt.pth --img_folder /mnt/nas/share-map/common/DMCar/rawdata/20210705024852/images_00021
# python3 utils/infer_onnx_model.py --save --model_name ocrnet_trt.DDRNet23_Slim.onnx --fp16
# onnx2trt ocrnet_trt.DDRNet23_Slim.onnx -o ocrnet_trt.DDRNet23_Slim_onnx.engine -d 16
