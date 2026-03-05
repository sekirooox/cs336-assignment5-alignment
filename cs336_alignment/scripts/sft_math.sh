#!/bin/bash

# 显式定义要运行的配置文件列表
JSON_PATH=(
    # "cs336_alignment/configs/sft_128.json"
    # "cs336_alignment/configs/sft_256.json"
    # "cs336_alignment/configs/sft_512.json"
    # "cs336_alignment/configs/sft_1024.json"
    "cs336_alignment/configs/sft_math.json"
    "cs336_alignment/configs/sft_math_correct.json"
)

# 如果没有定义任何配置文件，给出提示
if [ ${#JSON_PATH[@]} -eq 0 ]; then
  echo "错误: 请在脚本中定义 JSON_PATH 数组"
  echo "示例:"
  echo 'JSON_PATH=('
  echo '    "cs336_alignment/configs/sft1.json"'
  echo '    "cs336_alignment/configs/sft2.json"'
  echo ')'
  exit 1
fi

echo "=============================================="
echo "将按顺序运行以下配置文件:"
for config in "${JSON_PATH[@]}"; do
  echo "  - ${config}"
done
echo "=============================================="
echo ""

# 循环遍历 JSON_PATH 数组
for json_path in "${JSON_PATH[@]}"; do
  echo "=============================================="
  echo "开始训练，使用配置文件: ${json_path}"
  echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================="

  # 执行当前的训练脚本
  python cs336_alignment/train_sft.py --json_path "${json_path}"

  # 检查训练是否成功
  if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "训练成功完成: ${json_path}"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
  else
    echo "=============================================="
    echo "训练失败，配置文件: ${json_path}"
    echo "失败时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
    echo "中止后续任务。"
    exit 1
  fi
  echo ""  # 添加空行分隔不同训练任务
done

echo "=============================================="
echo "全部配置运行完成！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "运行的配置文件:"
for config in "${JSON_PATH[@]}"; do
  echo "  - ${config}"
done
echo "=============================================="