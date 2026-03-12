#!/bin/bash

# 检查是否传入了参数
if [ -z "$1" ]; then
  echo "用法: $0 <item1,item2,item3...>"
  echo "示例: $0 bottle,cm,door,lamp"
  exit 1
fi

# 将传入的第一个参数按逗号分割成数组
# IFS=, 设置内部字段分隔符为逗号
# read -ra 将读取的内容分割并存入数组
IFS=',' read -ra items <<< "$1"

echo "开始执行任务，解析到 ${#items[@]} 个物品..."

for item in "${items[@]}"; do
  # 去除可能存在的首尾空格 (可选，视具体情况而定)
  item=$(echo "$item" | xargs) 
  
  echo "--------------------------------"
  echo "正在处理: $item"
  
  sh scripts/${item}/collect_${item}_manip.sh
  
  if [ $? -ne 0 ]; then
    echo "错误：命令在 '$item' 上失败。"
    exit 1
  fi
done
