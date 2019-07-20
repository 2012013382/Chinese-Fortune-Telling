import tkinter as tk
from tkinter import *
import tkinter.filedialog
from feature_extract import feature_extract
from PIL import Image, ImageTk
import pickle
from lwz.cp_match.dist import best_match
from suanming.getname import xingmingceping
import os

#---------函数定义-------------
def choose_fiel():
	selectFileName = tk.filedialog.askopenfilename(title='选择文件')  # 选择文件
	e.set(selectFileName)
def upload_func(file_path, name_str):
	print(name_str)
	cand, face_score = feature_extract(file_path)
	cand = cand[0]
	match_idx = matcher.get_matched_for_male(cand)
	matched_path = os.path.join("photos",matcher.female[match_idx])
	# ---------颜值显示--------------------
	text1 = Text(windows)
	text1.insert("insert", str(face_score))
	text1.pack()

	family_name = name_str[0]
	first_name = name_str[1:]
	res_arr = xingmingceping(family_name,first_name)
	res = ""
	for i in res_arr:
		res += i + "\n"

	text2 = Text(windows)
	text2.insert("insert", res)
	text2.pack()
	showImg(matched_path)
	pass
def showImg(img1):
	load = Image.open(img1)
	render = ImageTk.PhotoImage(load)

	img = tkinter.Label(image=render)
	img.image = render
	img.place(x=200, y=100)
	pass
#---------窗口-----------------
matcher = best_match()
del matcher.female[16]
del matcher.female_cp_base[16]
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