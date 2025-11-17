task=${1:-caption}
dataset=${2:-coco2017}
device_num=${3:-1}
lever_lm=${4:-query_img_icd_img_text}
sampler=${5:-rand_sampler}

# 将 sampler 转换为大驼峰格式（与实际生成的文件名匹配）
case "$sampler" in
    rand_sampler)
        sampler_name="RandSampler"
        ;;
    text_sim_sampler)
        sampler_name="TextSimSampler"
        ;;
    img_sim_sampler)
        sampler_name="ImgSimSampler"
        ;;
    mix_sampler)
        sampler_name="MixSampler"
        ;;
    *)
        # 如果已经是大驼峰格式，直接使用
        sampler_name="${sampler}"
        ;;
esac

# 根据数据集自动设置 sample_num、model_name 和 dataset_name
case "$dataset" in
    okvqa*|OKVQA*)
        sample_num=800
        model_name="flamingo_3B"
        dataset_name="okvqa"
        ;;
    vqav2*|VQAV2*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="vqav2"
        ;;
    coco2017*|COCO2017*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="coco2017"
        ;;
    coco2014*|COCO2014*)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="coco2014"
        ;;
    *)
        sample_num=5000
        model_name="flamingo_3B"
        dataset_name="${dataset}"
        ;;
esac

# 构建数据文件名（使用 sampler_name 大驼峰格式）
data_file="${task}-${dataset_name}-${model_name}-${sampler_name}-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:${sample_num}.json"

# 构建检查点保存路径和文件名
# 完整的绝对路径（相对于项目根目录）
checkpoint_dir="./results/${dataset_name}/model_cpk"
# 文件名格式：模型_采样器_scorer_construct_order_beam_size_few_shot_candidate_num_sample_num
checkpoint_filename="${model_name}_${sampler_name}_infoscore_left_beam5_shot2_cand64_sample${sample_num}"

echo "Using data file: ${data_file}"
echo "Checkpoint will save to: ${checkpoint_dir}/${checkpoint_filename}_best.ckpt"

run_train() {
    local data_file=$1
 
    # 在 ex_name 中包含采样器名称，避免不同采样器互相覆盖
    local ex_name_prefix="main_${task}_${sampler_name}"

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        python train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        dirpath="${checkpoint_dir}" \
                        +checkpoint_filename="${checkpoint_filename}" \
                        trainer_args.devices=${device_num} \
                        dataset=${dataset} \
                        task=${task} \
                        +use_wandb=false \
                        +use_simple_logger=true 2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        python train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        trainer_args.max_epochs=20 \
                        trainer_args.val_check_interval=0.25 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        trainer_args.devices=${device_num} \
                        dataset=${dataset} \
                        task=${task} \
                        train.lever_lm.norm=false \
                        train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                        train.lever_lm.adapter=true \
                        +use_wandb=false \
                        +use_simple_logger=true 2>&1 | grep -v "UserWarning\|FutureWarning\|DeprecationWarning\|Found.*module.*eval mode"
    fi
}

run_train "${data_file}"
