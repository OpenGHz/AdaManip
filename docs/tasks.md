# Tasks

1. Bottle: Grasp the Cap. Rotate Cap after a failed Lift Up. Randomly sample Rotate/Lift if the previous action is Rotate.

2. Pen: Same as Bottle.

3. Pressure Cooker: Grasp the Handle. Rotate Handle after a failed Lift Up. Randomly sample Rotate/Lift if the previous action is Rotate.

4. Coffee Maker: Grasp Portafilter. Rotate Portafilter after a failed Pull Down. Randomly sample Rotate/Pull if the previous action is Rotate.

5. Window: Grasp the Handle. Randomly choose a direction to Rotate Handle. If failed, choose the other direction. Rotate Handle after a failed open trial. Randomly sample Rotate/Open if the previous action is Rotate.

6. Door: Same as Window.

7. Lamp: Randomly choose to Push/Clockwise Rotate/Counter Clockwise Rotate. Never choose a failed action.

8. Safe: Pull Door. If succeed, then continue opening the door. If failed, Rotate Knob. Randomly choose a direction to Rotate and choose the other one if failed. Then Pull Door again to open it.

9. Microwave: Pull Door. If succeed, then continue opening the door. If failed, Push Button. Then Pull Door again to open it.

## OPERATIONS AND SEMANTICS

<!-- 
上面是几个任务的操作说明，每个任务可能有多种实现路径，请为每个任务总结一个模糊命令和所有可能的操作集合（每个操作必须同时包含动作和对象，例如不能只是“向上提起”，必须是“向上提起瓶盖”），然后将该集合中的操作按不同的初始情况下存在的可能最少操作步骤形成最优具体无冗余步骤链集合。例如对于Window任务，总结如下：
- 模糊命令：打开窗户
- 操作集合：顺时针旋转把手、逆时针旋转把手、拉开窗户
- 无冗余步骤链：
  - 拉开窗户
  - 顺时针旋转把手 -> 拉开窗户
  - 逆时针旋转把手 -> 拉开窗户
如果一个操作最多可重复N次，则写成“N×操作名”，例如对于Bottle任务，总结如下：
- 模糊命令：打开瓶子
- 操作集合：旋转瓶盖、向上提起瓶盖
- 无冗余步骤链：
  - 向上提起瓶盖
  - N×旋转瓶盖 -> 向上提起瓶盖
一些注意事项：
- pull和push的方向未必分别是向下和向上，具体要根据实际情况来描述，如果不能确定方向，可以直接描述为“拉/推”，不要猜测方向造成误导。
-->

基于您提供的任务操作说明，我为您总结了每个任务的模糊命令、操作集合以及最优无冗余步骤链。

### 1. Bottle (瓶子)
- **模糊命令**：打开瓶子
- **操作集合**：旋转瓶盖、向上提起瓶盖
- **无冗余步骤链**：
  - 向上提起瓶盖
  - N×旋转瓶盖 -> 向上提起瓶盖

### 2. Pen (笔)
- **模糊命令**：打开笔
- **操作集合**：旋转笔盖、向上提起笔盖
- **无冗余步骤链**：
  - 向上提起笔盖
  - N×旋转笔盖 -> 向上提起笔盖

### 3. Pressure Cooker (高压锅)
- **模糊命令**：打开高压锅
- **操作集合**：旋转把手、向上提起把手
- **无冗余步骤链**：
  - 向上提起把手
  - N×旋转把手 -> 向上提起把手

### 4. Coffee Maker (咖啡机)
- **模糊命令**：拆卸咖啡机手柄
- **操作集合**：旋转手柄、向下拉动手柄
- **无冗余步骤链**：
  - 向下拉动手柄
  - N×旋转手柄 -> 向下拉动手柄

### 5. Window (窗户)
- **模糊命令**：打开窗户
- **操作集合**：顺时针旋转把手、逆时针旋转把手、拉开窗户
- **无冗余步骤链**：
  - 拉开窗户
  - 顺时针旋转把手 -> 拉开窗户
  - 逆时针旋转把手 -> 拉开窗户

### 6. Door (门)
- **模糊命令**：打开门
- **操作集合**：顺时针旋转把手、逆时针旋转把手、拉开门
- **无冗余步骤链**：
  - 拉开门
  - 顺时针旋转把手 -> 拉开门
  - 逆时针旋转把手 -> 拉开门

### 7. Lamp (台灯)
- **模糊命令**：打开/调节台灯
- **操作集合**：推开关、顺时针旋转开关、逆时针旋转开关
- **无冗余步骤链**：
  - 推开关
  - 顺时针旋转开关
  - 逆时针旋转开关
*(注：根据规则“Never choose a failed action”，这意味着一旦尝试某种动作成功即停止，若失败则不再重复该动作。因此不存在“先试A失败再试A”的路径，只有三种独立的单次尝试路径。)*

### 8. Safe (保险箱)
- **模糊命令**：打开保险箱
- **操作集合**：拉门、顺时针旋转旋钮、逆时针旋转旋钮
- **无冗余步骤链**：
  - 拉门
  - 顺时针旋转旋钮 -> 拉门
  - 逆时针旋转旋钮 -> 拉门
*(注：逻辑是“拉门”若失败，则必须“旋转旋钮”（若第一次旋转方向错误需换另一个方向），然后再“拉门”。因此核心路径是“旋转->拉”。)*

### 9. Microwave (微波炉)
- **模糊命令**：打开微波炉
- **操作集合**：拉门、按按钮
- **无冗余步骤链**：
  - 拉门
  - 按按钮 -> 拉门
