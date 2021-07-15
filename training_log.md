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
ddrnet23_slim_ocr(use above weights w/o retrain) | 1024(1.0) | 59.44
ddrnet23_slim | 1024(1.0) | 0.5391523
ddrnet23_slim_augment |  1024(1.0) | 0.5040767
