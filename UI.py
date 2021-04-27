import tkinter as tk

# creating main tkinter window/toplevel 
master = tk.Tk() 
master.config(height=5000, width=5000)
W="e"
# this will create a label widget 
l1 = tk.Label(master, text = "Height",width=10) 
l2 = tk.Label(master, text = "Width",width=10) 
  
# grid method to arrange labels in respective 
# rows and columns as specified 
l1.grid(row = 0, column = 0, sticky = W, pady = 3) 
l2.grid(row = 1, column = 0, sticky = W, pady = 3) 
  
# Input 
i1 = tk.Text(master,height=3,width=30) 
i2 =tk.Text(master,height=3,width=30) 
  
i1.grid(row = 0, column = 1, pady =3) 
i2.grid(row = 1, column = 1, pady = 3) 
  


#Out put


o1=tk.Text(master,height=5,width=65,bg='light cyan')
o1.grid(row = 0, column = 4, padx = 30,pady=30) 

o2=tk.Text(master,height=15,width=65,bg='light cyan')
o2.grid(row = 1, column = 4, padx = 30,pady=30) 



# checkbutton widget 
# c1 = tk.Checkbutton(master, text = "Preserve") 
# c1.grid(row = 2, column = 0, sticky = W, columnspan = 2) 
  
# adding image (remember image should be PNG and not JPG) 
# img = PhotoImage(file = r"C:\Users\Admin\Pictures\capture1.png") 
# img1 = img.subsample(2, 2) 
  
# setting image with the help of label 
# tk.Label(master, image = img1).grid(row = 0, column = 2, 
#        columnspan = 2, rowspan = 2, padx = 5, pady = 5) 
  
# button widget 
b1 = tk.Button(master, text = "Zoom in",height=2,width=20) 
b2 = tk.Button(master, text = "Zoom out",height=2,width=20) 
  
# arranging button widgets 
b1.grid(row = 0, column = 2, sticky = W, padx = 30,pady=30) 
b2.grid(row = 1, column = 2, sticky = W, padx = 30,pady=30) 
b1.config(anchor='center')



def print_text():
    text=i1.get("1.0",tk.END)
    o1.insert(tk.INSERT,text)
b1['command']=print_text
# infinite loop which can be terminated  
# by keyboard or mouse interrupt 
# print_text()
tk.mainloop() 