#!/bin/bash

# 检查是否传入了参数
if [ -z "$1" ]; then
  echo "用法: $0 <item1,item2,item3...>"
  echo "示例: $0 bottle,cm,door,lamp,microwave,pc,pen,safe,window"
  exit 1
fi

# 将传入的第一个参数按逗号分割成数组
IFS=',' read -ra items <<< "$1"

echo "开始执行任务，解析到 ${#items[@]} 个物品..."

role=$2

for item in "${items[@]}"; do
  # 去除可能存在的首尾空格
  item=$(echo "$item" | xargs) 
  
  echo "--------------------------------"
  echo "正在处理: $item"
  
  # 构造脚本路径
  script_path="scripts/${item}/collect_${item}_${role}.sh"
  
  # 【修改点】检查文件是否存在，如果不存在则跳过
  if [ ! -f "$script_path" ]; then
    echo "警告：脚本 '$script_path' 不存在，已跳过。"
    continue
  fi
  
  # 执行脚本
  sh "$script_path"
  
  # 检查执行结果（只有当脚本存在并执行后才检查错误）
  if [ $? -ne 0 ]; then
    echo "错误：命令在 '$item' 上执行失败。"
    exit 1
  fi
done
