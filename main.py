import sys
import multiprocessing
from PyQt6.QtWidgets import QApplication
from ui.main_window import CyberApp

# 必须保留的 Windows 多进程支持
if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)

    # 设置全局样式或字体（可选）
    # app.setStyle("Fusion")

    window = CyberApp()
    window.show()
    sys.exit(app.exec())