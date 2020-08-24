
# CATCaller
CATCaller: An end2end ONT Basecaller using Convolution-augmented Transformer.

#### DNA basecalling command
```angular2
bash run_caller.sh <model file> <fast5 folder> <signal window length> <basecalled_dir>
```
command parameters  
`model file`: we provide the `model/model.2048.chkpt` trained on Klebsiella pneumoniae genome.   
`fast5 folder`: directory of original sequencing fast5 files.   
`signal window length`: the length of the signal segment, default: `2048`.   
`basecalled_dir`: the output directory of basecalled files.  

#### Requirements and Installation
* Python version >= 3.6 
* CUDA/10.0
* pytorch_gpu version >= 1.2.0    
* [ctcdecode](https://github.com/parlance/ctcdecode.git)
* [dynamicConv/lightweightConv](https://github.com/pytorch/fairseq).
To install dynamicConv/lightweightConv module:
```
cd dynamicconv_layer
python cuda_function_gen.py
python setup.py build
python setup.py install --user
```
If you want to process new datasets and parallelly train a new model, the following tools are also needed:
* [apex](https://github.com/NVIDIA/apex)
* [tombo](https://github.com/nanoporetech/tombo)