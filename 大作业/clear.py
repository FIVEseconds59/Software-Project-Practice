import os

def rename_images(folder_path, prefix):
    # 获取文件夹中的所有文件，并按文件名排序
    files = sorted(os.listdir(folder_path))
    # 遍历文件并重命名
    for index, filename in enumerate(files):
        # 生成新的文件名，使用5位数补零格式
        new_name = f'{prefix}-{str(index+1).zfill(5)}.jpg'
        # 获取旧文件的完整路径
        old_file = os.path.join(folder_path, filename)
        # 获取新文件的完整路径
        new_file = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')

def process_folders(root_path, start, end):
    for folder_num in range(start, end + 1):
        folder_111 = os.path.join(root_path, f'{folder_num}/111')
        folder_112 = os.path.join(root_path, f'{folder_num}/112')
        if os.path.exists(folder_111):
            rename_images(folder_111, f'{folder_num}-111')
        if os.path.exists(folder_112):
            rename_images(folder_112, f'{folder_num}-112')

# 设置根目录
root_directory = 'dataset'  # 修改为实际路径

# 处理18到39文件夹
process_folders(root_directory, 18, 39)

print("所有图片重命名完成")
