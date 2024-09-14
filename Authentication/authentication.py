from tkinter import messagebox
import tkinter as tk
import sqlite3
# Create a connection to the database
cconn = sqlite3.connect('users.db')
cursor = cconn.cursor()
root = tk.Tk()
cursor.execute('''
                  CREATE TABLE IF NOT EXISTS users (
                      UserID INTEGER PRIMARY KEY AUTOINCREMENT,
                      Username TEXT UNIQUE NOT NULL,
                      Password TEXT NOT NULL
                  )
              ''')
#Execute the query
cconn.commit()


root.geometry("400x200")
root.title("Login Page")
username_label = tk.Label(root, text="Enter Username: ")
username_label.grid(row=0, column=0)
password_label = tk.Label(root, text="Enter Password: ")
password_label.grid(row=1, column=0)
e1 = tk.StringVar()
e2 = tk.StringVar()
username_entry = tk.Entry(root, textvariable=e1)
username_entry.grid(row=0, column=1)
password_entry = tk.Entry(root, textvariable=e2)
password_entry = tk.Entry(root, textvariable=e2, show = '*')
password_entry.grid(row=1, column=1)
users = {}
def login_toggle_password():
    # Check the current state of the password visibility
    if password_entry.cget('show') == '*':
        # If currently hidden, show the password
        password_entry.config(show='')
        showpassword_button.config(text='Hide Password')
    else:
        password_entry.config(show='*')
        showpassword_button.config(text='Show Password')

login_button =  tk.Button(
    root, 
    text="Login", 
    command=lambda: login(username_entry.get(), password_entry.get())
)
login_button.grid(row=4, column=0)
showpassword_button = tk.Button(root, text='Show Password', command=login_toggle_password)
showpassword_button.grid(row=1, column=2)
def login(username,password):
            cursor.execute('SELECT Password FROM users WHERE Username = ?', (username,))
            result = cursor.fetchone()

            if result and result[0] == password:
                messagebox.showinfo("Login", f"Welcome User {username}")
            else:
                messagebox.askretrycancel("Login", "Username or Password is incorrect. Try again?")


def create_new(username,password):
        try:
            cursor.execute('INSERT INTO users (Username, Password) VALUES (?, ?)', (username, password))
            cconn.commit()
            messagebox.showinfo("Account Creation", "Account Created Successfully")
        except sqlite3.IntegrityError:
            messagebox.showerror("Account Creation", "Username already exists")



def new_account():
    box = tk.Tk()
    box.geometry("400x200")
    box.title("New account")
    
    newusername_label = tk.Label(box, text="Enter Username: ")
    newusername_label.grid(row=0, column=0)
    e4 = tk.StringVar()
    e3 = tk.StringVar()
    newusername_entry = tk.Entry(box, textvariable=e3)
    newusername_entry.grid(row=0, column=1)
    quit_button = tk.Button(box, text="Quit",command = box.destroy)
    quit_button.grid(row=5, column=0)
    def create_toggle_password():
        # Check the current state of the password visibility
        if newpassword_entry.cget('show') == '*':
            # If currently hidden, show the password
            newpassword_entry.config(show='')
            showpassword_button.config(text='Hide Password')
        else:
            newpassword_entry.config(show='*')
            showpassword_button.config(text='Show Password')
    showpassword_button = tk.Button(box, text='Show Password', command=create_toggle_password)
    showpassword_button.grid(row=3, column=1)
    create_username_button = tk.Button(
        box, 
        text="Confirm Username and Password", 
        command=lambda: create_new(newusername_entry.get(),newpassword_entry.get())
    )
    create_username_button.grid(row=4, column=0)
    newpassword_label = tk.Label(box, text="Enter Password: ")

    newpassword_label.grid(row=2, column=0)

    newpassword_entry = tk.Entry(box, textvariable=e4)
    newpassword_entry = tk.Entry(box, textvariable=e4, show='*')
    newpassword_entry.grid(row=2, column=1)

    box.mainloop()

create_new_button = tk.Button(root, text="Create Account", command= new_account)
create_new_button.grid(row=4,column=1)

root.mainloop()
