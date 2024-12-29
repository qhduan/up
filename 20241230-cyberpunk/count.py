#!/usr/bin/env python3
import os
import sys
import unicodedata

def is_cjk(char):
    """判断一个字符是否是CJK字符"""
    code = ord(char)
    # CJK统一表意文字
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # CJK统一表意文字扩展A
    if 0x3400 <= code <= 0x4DBF:
        return True
    # CJK统一表意文字扩展B
    if 0x20000 <= code <= 0x2A6DF:
        return True
    # CJK统一表意文字扩展C
    if 0x2A700 <= code <= 0x2B73F:
        return True
    # CJK统一表意文字扩展D
    if 0x2B740 <= code <= 0x2B81F:
        return True
    # CJK统一表意文字扩展E
    if 0x2B820 <= code <= 0x2CEAF:
        return True
    return False

def count_cjk_in_file(filepath):
    """统计单个文件中的CJK字符数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return sum(1 for char in content if is_cjk(char))
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return 0

def count_cjk_in_directory(directory):
    """递归统计目录中所有文件的CJK字符数"""
    total_count = 0
    file_counts = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md') or file.endswith('.txt'):  # 只处理.md和.txt文件
                filepath = os.path.join(root, file)
                count = count_cjk_in_file(filepath)
                if count > 0:
                    relative_path = os.path.relpath(filepath, directory)
                    file_counts.append((relative_path, count))
                    total_count += count

    # 打印每个文件的统计结果
    if file_counts:
        print("\n文件统计结果:")
        for filepath, count in sorted(file_counts, key=lambda x: x[1], reverse=True):
            print(f"{filepath}: {count} 字")
        print(f"\n总计: {total_count} 字")
    else:
        print("未找到包含CJK字符的文件")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python count.py <目录路径>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是一个有效的目录")
        sys.exit(1)

    count_cjk_in_directory(directory)

if __name__ == "__main__":
    main()
