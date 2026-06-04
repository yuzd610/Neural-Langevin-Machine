import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

def plot_eigenvalue_spectra_by_size(input_dir='checkpoints_by_size', output_dir='eigenvalue_plots_by_size'):
    """
    Load weight matrices (J), compute their eigenvalues, and plot them in the complex plane.
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found. Please check the path.")
        return

    # 筛选出 J 矩阵权重文件 (.npy)
    files = os.listdir(input_dir)
    j_files = [f for f in files if f.startswith('J_size_') and f.endswith('.npy')]

    if not j_files:
        print(f"No J matrix weight files found in '{input_dir}'.")
        return

    # 辅助函数：提取文件名中的 size 和 run 以便排序
    def extract_info(filename):
        match = re.search(r'J_size_(\d+)_run_(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    # 按照数据集大小和 run_idx 进行排序
    j_files.sort(key=extract_info)

    print(f"Found {len(j_files)} J-matrix files. Starting eigenvalue calculation and plotting...")

    for filename in tqdm(j_files, desc="Processing"):
        filepath = os.path.join(input_dir, filename)
        
        # 提取 metadata (size 和 run) 用于命名图表和设定 title
        match = re.search(r'J_size_(\d+)_run_(\d+)', filename)
        if not match:
            continue
            
        size = match.group(1)
        run = match.group(2)

        # 1. 载入权重矩阵 J
        J = np.load(filepath)

        # 2. 计算本征值
        eigenvalues = np.linalg.eigvals(J)

        # 3. 提取实部和虚部
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        # 4. 可视化绘图
        plt.figure(figsize=(8, 8))
        
        # 绘制本征值散点图（使用与原代码相同的大小和透明度以观察密度分布）
        plt.scatter(real_parts, imag_parts, s=45, alpha=0.6, color='blue', edgecolors='none')

        # 添加坐标轴参考线
        plt.axhline(0, color='black', linewidth=3, linestyle='--')
        plt.axvline(0, color='black', linewidth=3, linestyle='--')
        
        # 图表样式设置
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlabel('Re', fontsize=32)
        plt.ylabel('Im', fontsize=32)
        
        # 更新 Title 格式为 N_data=xxx
        # 注意这里使用了 Matplotlib 的 MathText 语法，双大括号 {{}} 是为了在 f-string 中转义
        plt.title(f'$N_{{data}}={size}$', fontsize=32)
        
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # 关键设置：1:1 的纵横比，确保椭圆律的几何形状不被拉伸变形
        plt.axis('equal') 

        # 5. 保存图片
        # 将不同的 size 和 run 加入到生成图片的文件名中，防止互相覆盖
        save_filename = f'eigen_spectrum_size_{size}_run_{run}.pdf'
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # 关闭画布，释放内存

    print(f"\nProcessing complete! Plots are saved in the '{output_dir}' folder.")

if __name__ == '__main__':
    plot_eigenvalue_spectra_by_size()