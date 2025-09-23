from pathlib import Path
import cv2
import argparse

# 添加项目根目录到 Python 路径
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.debayer import debayer_image

def main():
    parser = argparse.ArgumentParser(description="Batch debayer images with even-image flip rule.")
    parser.add_argument("--input-dir", type=str, default="./data/raw", help="Input directory containing raw PNGs")
    parser.add_argument("--output-dir", type=str, default="./output/rgb", help="Output directory for processed PNGs")
    parser.add_argument("--pattern", type=str, default="GBRG", choices=["BGGR", "GBRG", "GRBG", "RGGB"], help="Bayer pattern")
    args = parser.parse_args()

    # 输入输出文件夹
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pattern = args.pattern

    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有PNG文件并按名称排序
    png_files = sorted(list(input_dir.glob("*.png")))

    if not png_files:
        print("没有找到PNG文件")
        return

    # 批量处理每个PNG文件
    for idx, png_file in enumerate(png_files, 1):
        try:
            # 构建输出文件路径
            output_path = output_dir / png_file.name
            png_file_name = int(png_file.name[:-4])
            # 奇数图片进行解拜耳处理
            if png_file_name % 2 == 1:  # 奇数
                print(f"处理奇数图片: {png_file_name}")
                debayer_image(str(png_file), str(output_path), pattern)
            else:  # 偶数
                print(f"水平翻转偶数图片: {png_file_name}")
                img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
                flipped = cv2.flip(img, 1)
                cv2.imwrite(str(output_path), flipped)
                
            print(f"完成: {output_path}")
        except Exception as e:
            print(f"处理 {png_file} 失败: {str(e)}")


if __name__ == "__main__":
    main()

