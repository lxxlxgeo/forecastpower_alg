import os
import bz2
from glob import glob
from tqdm import tqdm

def decompress_files(input_dir, output_dir):
    """
    解压输入目录中的所有 .bz2 文件，并将解压后的文件保存到输出目录，文件后缀改为 .grib2。

    参数:
        input_dir (str): 输入目录，包含待解压的 .bz2 文件。
        output_dir (str): 输出目录，用于保存解压后的 .grib2 文件。
    """
    # 检查输出目录是否存在，不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 glob 查找所有 .bz2 文件
    bz2_files = glob(os.path.join(input_dir, '*.bz2'))
    
    # 解压每个文件并显示进度条
    for bz2_file in tqdm(bz2_files, desc="Decompressing files", unit="file"):
        # 获取文件名并设置输出路径
        filename = os.path.basename(bz2_file)
        output_filename = filename.replace('.bz2', '.grib2')
        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            print('文件已解压!!!')
            continue
        # 解压文件并保存
        with bz2.open(bz2_file, 'rb') as bz_file:
            decompressed_data = bz_file.read()
        
        with open(output_path, 'wb') as output_file:
            output_file.write(decompressed_data)
    
    print("所有文件解压完成。")