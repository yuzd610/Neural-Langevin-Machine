import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. Configuration Parameters
# ==========================================
input_dir = 'generated_trajectories'
output_combined_dir = 'combined_visualizations'
os.makedirs(output_combined_dir, exist_ok=True)

# 目标模型的训练迭代步 (t_age)
target_model_step = 23836
# 我们希望可视化的采样时间步 (Langevin dynamics steps)
target_traj_steps = [20, 99, 500,2530, 12808]
img_h, img_w = 28, 28


def generate_evolution_plot():
    # --- 1. 查找目标模型文件 ---
    all_files = os.listdir(input_dir)
    # 寻找以 _step_23836.npy 结尾的文件 (请确保你的文件名格式匹配)
    match = [f for f in all_files if f.endswith(f'_step_{target_model_step}.npy')]

    if not match:
        print(f"没有找到模型 {target_model_step} 的对应文件！请检查 input_dir 或文件名。")
        return

    file_name = match[0]
    file_path = os.path.join(input_dir, file_name)

    # --- 2. 加载数据 ---
    print(f"正在读取模型文件: {file_name}")
    data = np.load(file_path, allow_pickle=True).item()

    # 提取保存的步数列表和轨迹张量
    saved_steps = list(data['steps'])
    trajectories = data['trajectories']

    num_rows = len(target_traj_steps)

    # --- 3. 布局计算 ---
    fig_width = 12
    model_block_height = 2
    gap_height = 0.05
    fig_height = (model_block_height + gap_height) * num_rows

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 单列布局，左侧留出空间放置文字
    outer_grid = gridspec.GridSpec(
        num_rows, 1,
        hspace=gap_height / model_block_height,
        left=0.12, right=0.98, top=0.95, bottom=0.05
    )

    # --- 4. 遍历指定的采样时间步并绘图 ---
    for idx, traj_step in enumerate(target_traj_steps):
        try:
            # 找到目标时间步在保存列表中的索引
            if traj_step not in saved_steps:
                print(f"警告: 采样步 {traj_step} 不在保存的时间步列表中 {saved_steps}")
                continue

            step_index = saved_steps.index(traj_step)

            # 提取该时间步对应的图像数据并转置：shape [batch_size, 784]
            images = trajectories[step_index].T

            # 排列 2x10 的图像网格 (提取第 90 到 109 个样本)
            samples = images[90:110].reshape(-1, img_h, img_w)
            row1 = np.concatenate(samples[0:10], axis=1)
            row2 = np.concatenate(samples[10:20], axis=1)
            combined_img = np.concatenate([row1, row2], axis=0)

            # 创建子图
            ax_img = fig.add_subplot(outer_grid[idx, 0])
            ax_img.imshow(combined_img, cmap='gray', aspect='equal', interpolation='nearest')
            ax_img.axis('off')

            # --- 文字标签放置 ---
            y_center = combined_img.shape[0] / 2.0
            x_offset = -8  # 负数将文字向左偏移，使其不遮挡图像

            ax_img.text(x_offset, y_center, str(traj_step),
                        fontsize=34,
                        fontweight='bold',
                        color='#333333',  # 使用深灰色/黑色，不再使用渐变
                        ha='right',
                        va='center',
                        clip_on=False)

        except Exception as e:
            print(f"处理采样步 {traj_step} 时发生错误: {e}")

    # --- 5. 保存并展示结果 ---
    save_filename = f"evolution_t_age_{target_model_step}.pdf"
    save_path = os.path.join(output_combined_dir, save_filename)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
    print(f"\n可视化完成！图像已保存至: {save_path}")


if __name__ == "__main__":
    generate_evolution_plot()