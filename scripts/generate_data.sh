task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
sampler=${4:-rand_sampler}
beam_model=${5:-flamingo_3B}

# 根据数据集自动设置 sample_num
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800  # OKVQA 数据集较小(9k)，使用 800
        ;;
    vqav2*|VQAV2*|vqa*)
        sample_num=5000  # VQAv2 数据集大(443k)，使用 5000
        ;;
    coco*|COCO*)
        sample_num=5000  # COCO 数据集使用 5000
        ;;
    *)
        sample_num=5000  # 默认使用 5000
        ;;
esac

echo "Dataset: $dataset, Sampler: $sampler, Beam Model: $beam_model, Sample num: $sample_num"

python generate_data.py beam_size=5 \
                        cand_num=64 \
                        sample_num=${sample_num} \
                        gpu_ids="${gpu_ids}" \
                        task=${task} \
                        dataset=${dataset} \
                        sampler=${sampler} \
                        infer_model=${beam_model} \
                        few_shot_num=2