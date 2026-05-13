"""
Real-ESRGAN 超分推理脚本
用法: python run.py --input <图片路径或文件夹> [--scale 4] [--output <输出目录>]
"""
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-ESRGAN 图像超分辨率')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图片路径或文件夹路径')
    parser.add_argument('--output', '-o', type=str, default='./results', help='输出目录 (默认: ./results)')
    parser.add_argument('--scale', '-s', type=int, default=4, help='放大倍数: 2 或 4 (默认: 4)')
    parser.add_argument('--face', '-f', action='store_true', help='启用人脸增强')
    args = parser.parse_args()

    model_map = {
        2: 'RealESRGAN_x2plus',
        4: 'RealESRGAN_x4plus',
    }
    model = model_map.get(args.scale, 'RealESRGAN_x4plus')

    cmd = f'python inference_realesrgan.py -i "{args.input}" -o "{args.output}" -n {model}'
    if args.face:
        cmd += ' --face_enhance'

    print(f'执行: {cmd}')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.system(cmd)