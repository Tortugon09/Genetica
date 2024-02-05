import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
import perceptron as pl

def select_file():
    filename = filedialog.askopenfilename(initialdir="/", title="SELECCIONAR ARCHIVO (CSV)",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    file_label.config(text=filename)
    return filename

def start_training():
    learning_rate = float(learning_rate_entry.get())
    epochs = int(epochs_entry.get())
    file_path = file_label.cget("text")
    if file_path: 
        threading.Thread(target=lambda: pl.train_perceptron(learning_rate, epochs, file_path, progress_bar)).start()
    else:
        messagebox.showwarning("Advertencia", "Debe seleccionar un archivo CSV.")

def display_graphs():
    pl.display_results()

def generate_report():
    np.set_printoptions(precision=4, suppress=True)
    initial_weights, final_weights, epochs, error = pl.get_weights()
    report_content = (f"EPOCAS: {epochs}\n"
                      f"TASA DE APRENDIZAJE: {learning_rate_entry.get()}\n"
                      f"ERROR PERMISIBLE: {error}\n\n"
                      f"PESOS INICIALES DE W:\n{initial_weights}\n\n"
                      f"PESOS FINALES DE W:\n{final_weights}")
    report_text.config(state="normal")
    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, report_content)
    report_text.config(state="disabled")



root = tk.Tk()
root.title("Perceptron Training")
root.geometry('600x400') 
root.minsize(400, 750) 


style = ttk.Style()
style.theme_use('alt')

bg_color = "#f0f0f0"
text_color = "#333333"
font = ('Helvetica', 10)

style.configure("TProgressbar", thickness=20, troughcolor=bg_color, bordercolor=bg_color, background='#333333')



root.configure(bg=bg_color)

labelframe = ttk.LabelFrame(root, text="INGRESAR DATOS", padding="20 20 20 20")
labelframe.pack(fill="both", expand="yes", padx=20, pady=20)

progress_buttons_frame = ttk.LabelFrame(root, text="Controles", padding="10 10 10 10")
progress_buttons_frame.pack(fill="x", padx=20, pady=10)

report_frame = ttk.LabelFrame(root, text="REPORTE", padding="10 10 10 10")
report_frame.pack(fill="both", expand=True, padx=20, pady=10) 


report_text = tk.Text(report_frame, wrap="word", font=font, bg=bg_color, fg=text_color)
report_text.pack(expand=True, fill="both", padx=10, pady=10) 
report_text.config(state="disabled")  


ttk.Label(labelframe, text="TASA DE APRENDIZAJE:", foreground=text_color, font=font).pack(pady=5)
learning_rate_entry = ttk.Entry(labelframe, width=50)
learning_rate_entry.pack()

ttk.Label(labelframe, text="EPOCAS (ITERACIONES):", foreground=text_color, font=font).pack(pady=5)
epochs_entry = ttk.Entry(labelframe, width=50)
epochs_entry.pack()

file_frame = ttk.Frame(labelframe, padding="10 10 10 10")
file_frame.pack(fill="x", expand="yes")
ttk.Button(file_frame, text="Select CSV File", command=select_file).pack(side="left", padx=5)
file_label = ttk.Label(file_frame, text="", background=bg_color, foreground=text_color, font=font)
file_label.pack(side="left", fill="x", expand=True)

buttons_frame = ttk.Frame(labelframe)
buttons_frame.pack(pady=20)
ttk.Button(buttons_frame, text="INICIAR", command=start_training).pack(side="left", padx=5)
ttk.Button(buttons_frame, text="GRÁFICAS", command=display_graphs).pack(side="left", padx=5)
ttk.Button(labelframe, text="REPORTE", command=generate_report).pack(pady=10)

progress_bar = ttk.Progressbar(labelframe, orient="horizontal", mode="determinate", style="TProgressbar")
progress_bar.pack(fill="x", pady=10)

root.mainloop()
