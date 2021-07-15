# Xavier Nano Benchmark
The latency is calculated between transfering input to gpu and fetch results back on cpu.

Network| Resolution  | Latency (ms) | GFlops
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


