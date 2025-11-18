import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='数据集根目录路径')
    parser.add_argument('--dataset_type', type=str, default='auto', 
                        choices=['vqav2', 'okvqa', 'auto'],
                        help='数据集类型：vqav2（有v2_前缀）, okvqa（无前缀）, auto（自动检测）')
    args = parser.parse_args()
    root = Path(args.root_path)
    
    # 自动检测数据集类型
    if args.dataset_type == 'auto':
        # 检查是否存在v2_前缀的文件
        if (root / 'v2_OpenEnded_mscoco_train2014_questions.json').exists():
            dataset_type = 'vqav2'
            print(f"自动检测到数据集类型: VQAv2 (有v2_前缀)")
        elif (root / 'OpenEnded_mscoco_train2014_questions.json').exists():
            dataset_type = 'okvqa'
            print(f"自动检测到数据集类型: OKVQA (无v2_前缀)")
        else:
            raise FileNotFoundError(f"在 {root} 中未找到VQAv2或OKVQA格式的问题文件")
    else:
        dataset_type = args.dataset_type
        print(f"使用指定的数据集类型: {dataset_type}")
    
    # 根据数据集类型设置文件名
    if dataset_type == 'vqav2':
        prefix = 'v2_'
        save_folder = 'vqav2_hf'
    else:  # okvqa
        prefix = ''
        save_folder = 'okvqa_hf'
    
    train_ques = root / f'{prefix}OpenEnded_mscoco_train2014_questions.json'
    train_ann = root / f'{prefix}mscoco_train2014_annotations.json'
    val_ques = root / f'{prefix}OpenEnded_mscoco_val2014_questions.json'
    val_ann = root / f'{prefix}mscoco_val2014_annotations.json'

    save_path = root / save_folder
    if not save_path.exists():
        save_path.mkdir()
    
    print(f"\n读取文件:")
    print(f"  训练问题: {train_ques.name}")
    print(f"  训练标注: {train_ann.name}")
    print(f"  验证问题: {val_ques.name}")
    print(f"  验证标注: {val_ann.name}")
    print(f"  保存目录: {save_folder}\n")

    ques = json.load(open(train_ques))
    ann = json.load(open(train_ann))
    quesid2question = {}
    for q in ques['questions']:
        quesid2question[q['question_id']] = q['question']
    total_data = []
    for a in ann['annotations']:
        a['question'] = quesid2question[a['question_id']]
        total_data.append(a)
    ann['annotations'] = total_data
    with open(save_path / 'vqav2_mscoco_train2014.json', 'w') as f:
        json.dump(ann, f)

    ques = json.load(open(val_ques))
    ann = json.load(open(val_ann))
    quesid2question = {}
    for q in ques['questions']:
        quesid2question[q['question_id']] = q['question']
    total_data = []
    for a in ann['annotations']:
        a['question'] = quesid2question[a['question_id']]
        total_data.append(a)
    ann['annotations'] = total_data
    with open(save_path / 'vqav2_mscoco_val2014.json', 'w') as f:
        json.dump(ann, f)
