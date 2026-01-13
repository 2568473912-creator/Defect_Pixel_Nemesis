import sys
import multiprocessing
from PyQt6.QtWidgets import QApplication
from ui.main_window import CyberApp
import sys
import os

def get_resource_path(relative_path):
    """ 获取资源绝对路径，兼容开发环境和打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的临时目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 使用时：
# ini_path = get_resource_path("config.ini")
# 必须保留的 Windows 多进程支持
if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)

    # 设置全局样式或字体（可选）
    # app.setStyle("Fusion")

    window = CyberApp()
    window.show()
    sys.exit(app.exec())