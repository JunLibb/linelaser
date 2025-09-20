
from fscan2.debayer import debayer_image
from pathlib import Path
import cv2

# 输入输出文件夹
input_dir = Path("./img/calimages")
output_dir = Path("./output/rgb")

pattern = "GBRG"
# 创建输出文件夹
output_dir.mkdir(parents=True, exist_ok=True)

# 获取所有PNG文件并按名称排序
png_files = sorted(list(input_dir.glob("*.png")))

if not png_files:
    print("没有找到PNG文件")
    exit(1)

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

