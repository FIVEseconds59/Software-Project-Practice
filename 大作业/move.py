import os
import shutil


def rename_and_collect_images(folder_path, prefix, target_folder):
    # 获取文件夹中的所有文件，并按文件名排序
    files = sorted(os.listdir(folder_path))
    # 遍历文件并重命名
    for index, filename in enumerate(files):
        # 生成新的文件名，使用5位数补零格式
        new_name = f'{prefix}-{str(index + 1).zfill(5)}.jpg'
        # 获取旧文件的完整路径
        old_file = os.path.join(folder_path, filename)
        # 获取新文件的完整路径
        new_file = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')
        # 移动文件到目标文件夹
        shutil.move(new_file, os.path.join(target_folder, new_name))
        print(f'Moved: {new_file} -> {os.path.join(target_folder, new_name)}')


def process_folders(root_path, start, end, target_folder):
    # 创建目标文件夹，如果不存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for folder_num in range(start, end + 1):
        folder_111 = os.path.join(root_path, f'{folder_num}/111')
        folder_112 = os.path.join(root_path, f'{folder_num}/112')
        if os.path.exists(folder_111):
            rename_and_collect_images(folder_111, f'{folder_num}-111', target_folder)
        if os.path.exists(folder_112):
            rename_and_collect_images(folder_112, f'{folder_num}-112', target_folder)


# 设置根目录和目标文件夹路径
root_directory = 'dataset'  # 修改为实际路径
target_directory = 'data'  # 修改为目标data文件夹路径

# 处理18到39文件夹
process_folders(root_directory, 18, 39, target_directory)

print("所有图片重命名并移动完成")
