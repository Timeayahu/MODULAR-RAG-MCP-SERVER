


import argparse

def main():
    # 1. 创建解析器对象（设置程序的描述信息）
    parser = argparse.ArgumentParser(description="这是一个 RAG 系统的文档处理脚本")

    # 2. 添加参数（告诉程序该接收什么）
    # 位置参数（必填）
    parser.add_argument("input_file", help="要处理的文件路径")
    
    # 可选参数（带 --，有默认值）
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    
    # 布尔开关（如果输入了 --verbose，则为 True）
    parser.add_argument("--verbose", action="store_true", help="显示详细进度")

    # 3. 解析参数（将命令行输入的字符串转为 Python 对象）
    args = parser.parse_args()

    # 4. 使用参数
    print(f"正在处理: {args.input_file}")
    if args.verbose:
        print(f"当前批次设置为: {args.batch_size}")

if __name__ == "__main__":
    main()