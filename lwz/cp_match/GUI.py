import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

#---------函数定义-------------
def choose_fiel():
	selectFileName = tk.filedialog.askopenfilename(title='选择文件')  # 选择文件
	e.set(selectFileName)
def upload_func(file_path, name_str):
	showImg(file_path)
	print(name_str)
	pass
def showImg(img1):
	load = Image.open(img1)
	render = ImageTk.PhotoImage(load)

	img = tkinter.Label(image=render)
	img.image = render
	img.place(x=200, y=100)
	pass
#---------窗口-----------------
windows = tk.Tk()
e = tk.StringVar()
e_entry = tkinter.Entry(windows, width=68, textvariable=e)
e_entry.pack()
#---------上传图片button--------------
submit_button = tkinter.Button(windows, text ="选择文件", command = choose_fiel)
submit_button.pack()
#---------输入姓名button--------------
name_para = tk.StringVar()
name = tkinter.Entry(windows, textvariable = name_para)
name.pack()
#---------处理图片button--------------
submit_button = tkinter.Button(windows, text ="上传", command = lambda:upload_func(e_entry.get(), name.get()))
submit_button.pack()


windows.mainloop()