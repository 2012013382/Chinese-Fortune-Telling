# coding:utf-8

from PyQt5 import QtCore,QtGui,QtWidgets
import sys
import qtawesome

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(960,700)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout) # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget() # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout) # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget,0,0,12,2) # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget,0,2,12,10) # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget) # 设置窗口主部件

        self.left_close = QtWidgets.QPushButton("") # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("") # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_label_1 = QtWidgets.QPushButton("每日推荐")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("我的音乐")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.left_label_3.setObjectName('left_label')

        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_layout.addWidget(self.left_mini, 0, 0,1,1)
        self.left_layout.addWidget(self.left_close, 0, 2,1,1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1,1,0,1,3)
        self.left_layout.addWidget(self.left_label_2, 5, 0,1,3)
        self.left_layout.addWidget(self.left_label_3, 9, 0,1,3)

        self.right_bar_widget = QtWidgets.QWidget() # 右侧顶部搜索框部件
        self.right_bar_layout = QtWidgets.QGridLayout() # 右侧顶部搜索框网格布局
        self.right_bar_widget.setLayout(self.right_bar_layout)
        self.search_icon = QtWidgets.QLabel(chr(0xf002) + ' '+'搜索  ')
        self.search_icon.setFont(qtawesome.font('fa', 16))
        self.right_bar_widget_search_input = QtWidgets.QLineEdit()
        self.right_bar_widget_search_input.setPlaceholderText("输入歌手、歌曲或用户，回车进行搜索")
        self.right_bar_layout.addWidget(self.search_icon,0,0,1,1)
        self.right_bar_layout.addWidget(self.right_bar_widget_search_input,0,1,1,8)
        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 1, 9)

        self.left_close.setFixedSize(15,15) # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15) # 设置最小化按钮大小

        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
            QWidget#left_widget{
            background:gray;
            border-top:1px solid white;
            border-bottom:1px solid white;
            border-left:1px solid white;
            border-top-left-radius:10px;
            border-bottom-left-radius:10px;
            }
        ''')

        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
                }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                }
        ''')

        self.setWindowOpacity(0.9) # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) # 隐藏边框
        self.main_layout.setSpacing(0)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
