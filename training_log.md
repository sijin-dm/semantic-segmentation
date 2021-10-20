# Mapillary v1.2

Network| Resolution  | mIOU
|----|----|----|
HRNet-OCR-official | 2177 (1.0) | 0.548
HRNet-OCR-official | 960 (1.0) | 0.5643991
HRNet-OCR-official | 768 (1.0) | 0.54009104
HRNet-OCR-official | 960 (flip,0.5,1.0,2.0) | 0.5988845
HRNet-OCR-official | 768 (flip,0.5,1.0,2.0) | 0.58811766
HRNet-OCR-dm-train-gpu001-100epoch  | 2177 (1.0) | 0.484
HRNet-OCR-dm-train-gpu001-100epoch  | 960 (1.0) | 0.54237956
HRNet-OCR-dm-train-gpu001-100epoch  | 768 (1.0) | 0.52132976
HRNet-OCR-dm-train-gpu001-100epoch  | 2177 (flip,0.5,1.0,2.0) | 0.5780145
HRNet-OCR-dm-train-gpu001-100epoch  | 960 (flip,0.5,1.0,2.0) | 0.5777298
HRNet-OCR-dm-train-gpu001-100epoch  | 768 (flip,0.5,1.0,2.0) | 0.56806904


# Mapillary v2.0_dm
Network| Resolution  | mIOU
|----|----|----|
HRNet-OCR-dm-train | 2177 (1.0) | 0.6443699
HRNet-OCR-dm-train  | 960 (1.0) | 0.67196906
HRNet-OCR-dm-train  | 768 (1.0) | 0.66279215
HRNet-OCR-dm-train | 2177 (flip,0.5,1.0,2.0) |0.69360286
HRNet-OCR-dm-train  | 960 (flip,0.5,1.0,2.0) | 0.69965035
HRNet-OCR-dm-train  | 768 (flip,0.5,1.0,2.0) | 0.6981542
ddrnet23_slim_ocr                    | 1024 (0.5,1.0,2.0) | 0.6289
ddrnet23_slim_ocr(use above weights w/o retrain) | 1024(1.0) | 0.5944
ddrnet23_slim | 1024(1.0) | 0.5391523
ddrnet23_slim_augment |  1024(1.0) | 0.5040767
naive_lite_hrnet_ocr               | 1024 (0.5,1.0,2.0) | 0.5454124
lite_hrnet_ocr               | 1024 (0.5,1.0,2.0) | 0.5753439
HRNet-OCR-dm-train-wo-rmi-loss | 2177 (flip,0.5,1.0,2.0) |0.6662466

# Mapillary v2.0_dm distillation
Network| Resolution  | mIOU | Remark
|----|----|----|----|
OCR_HRNet_Mscale_OCR_DDRNET23_SLIM_Mscale | 1024 (0.5,1.0,2.0) | 0.61888266 | Full
OCR_HRNet_Mscale_OCR_DDRNET23_SLIM_Mscale | 1024 (0.5,1.0,2.0) | 0.62236065 | Full, MC Dropout
OCR_HRNet_Mscale_OCR_DDRNET23_SLIM_Mscale | 1024 (0.5,1.0,2.0) | 0.46392435 | 0.1 miniset
OCR_HRNet_Mscale_OCR_DDRNET23_SLIM_Mscale | 1024 (0.5,1.0,2.0) | 0.4978166 | 0.1 miniset,MC Dropout
