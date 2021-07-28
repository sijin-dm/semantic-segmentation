# Xavier Nano Benchmark
The latency is calculated between transfering input to gpu and fetch results back on cpu with fp16.

Network| Resolution  | Latency (ms) 
|----|----|----|
DDRNet23-Slim_TRT(Backbone) | 768x384 | 33  |
DDRNet23-Slim_TRT(Backbone)-Torch(Interpolate+Softmax+Max) | 768x384 | 54 
DDRNet23-Slim_TRT(Backbone)-Torch(Interpolate+Softmax) | 768x384 | 95 
DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax) | 768x384 | 100  
DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax)-Torch(Max) | 768x384 | 60  
OCR-DDRNet23-Slim_TRT(Backbone)| 768x384 | 78 ms
OCR-DDRNet23-Slim_TRT(Backbone)-Torch(Interpolate+Softmax)| 768x384 | 136  ms
OCR-DDRNet23-Slim_TRT(Backbone)-Torch(Interpolate+Softmax+Max)| 768x384 | 100  ms
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax)-Torch(Max)| 768x384 | 126  ms

# Xavier NX Benchmark
Network| Resolution  | Latency (ms) | GFlops | Mode 
|----|----|----|
OCR-HRNet_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 151  ms  | fp16
OCR-HRNet_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 109  ms  | int8
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 31  ms | fp16
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 29  ms | int8
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax)-Torch(Max)| 768x384 | 23  ms | fp16
OCR-DDRNet23-Slim_Mscale_0.5_1.0_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 42  ms | fp16
OCR-HRNet_W18_SMALL_V2_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 72  ms | fp16
OCR-HRNet_W18_SMALL_V2_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 61  ms | int8


# 3090 Benchmark
Network| Resolution  | Latency (ms) | mIOU
|----|----|----|
OCR-HRNet_Mscale_0.5_1.0_2.0_TRT(Backbone+Interpolate+Softmax+Max)| 1280x720 | 120  ms|
OCR-HRNet_Mscale_0.5_1.0_TRT(Backbone+Interpolate+Softmax+Max)| 1280x720 | 30  ms | 0.606
OCR-HRNet_Mscale_0.5_1.0_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 15  ms|
OCR-HRNet_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 10  ms |
OCR-HRNet_TRT(Backbone+Interpolate+Softmax)-Torch(Max)| 768x384 | 10  ms |
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax)-Torch(Max)| 768x384 | 2  ms |
OCR-DDRNet23-Slim_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 |  2  ms |
HRNet_lite_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 6  ms |
Naive_HRNet_lite_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 4  ms |
OCR-HRNet_W18_SMALL_V2_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 5  ms |
OCR-HRNet_lite_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 8  ms |
OCR-Naive_HRNet_lite_TRT(Backbone+Interpolate+Softmax+Max)| 768x384 | 6  ms |
