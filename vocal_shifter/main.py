import sys
from PyQt6.QtWidgets import QApplication
from .main_window import VocalShifterGUI

def main():
    app = QApplication(sys.argv)
    window = VocalShifterGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
