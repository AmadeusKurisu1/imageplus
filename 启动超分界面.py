"""Real-ESRGAN 图像超分辨率 - 双击启动 Web UI"""
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real-ESRGAN"))
sys.path.insert(0, os.getcwd())

import app

if __name__ == "__main__":
    print("启动 Real-ESRGAN Web UI...")
    app.demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)