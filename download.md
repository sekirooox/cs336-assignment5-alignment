# Model
+ Qwen 2.5 Math 1.5B Base (for reasoning experiments): /data/a5-alignment/models/Qwen2.5-Math-1.5B  
+ Llama 3.1 8B Base (for optional instruction tuning experiments): /data/a5-alignment/models/Llama-3.1-8B  
+ Llama 3.3 70B Instruct (for optional instruction tuning experiments): /data/a5-alignment/models/Llama-3.3-70B-Instruct

# Dataset
+ Math: nlile/hendrycks-MATH-benchmark (自行安装)
+ GSM8K Dataset

# Download Command

wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
sudo apt-get update
sudo apt-get install aria2
./hfd.sh Qwen/Qwen2.5-Math-1.5B --local-dir ./model/Qwen2.5-Math-1.5B
./hfd.sh nlile/hendrycks-MATH-benchmark --dataset --local-dir ./dataset/hendrycks-MATH-benchmark

