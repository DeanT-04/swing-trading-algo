import tkinter as tk
from tkinter import ttk

def main():
    # Create root window
    root = tk.Tk()
    root.title("Simple UI Test")
    root.geometry("400x300")
    
    # Create a label
    label = ttk.Label(root, text="UI Test Successful!", font=("Arial", 16))
    label.pack(pady=20)
    
    # Create a button
    button = ttk.Button(root, text="Click Me", command=lambda: label.config(text="Button Clicked!"))
    button.pack(pady=10)
    
    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
