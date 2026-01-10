import threading
import tkinter as tk
from tkinter import ttk, messagebox
import io
import sys
import os

# Import your existing agent function

from RsearchAgent import run_agent


class TextRedirector(io.TextIOBase):
    """Redirect print() output to a Tkinter Text widget."""
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, s: str):
        self.text_widget.after(0, lambda: (self.text_widget.insert(tk.END, s), self.text_widget.see(tk.END)))
        return len(s)

    def flush(self):
        pass


def resource_path(relative_path: str) -> str:
    """
    Ensures files (like memory.json) are written next to the executable.
    When packaged, sys._MEIPASS exists; but we want writable folder = exe dir.
    """
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def run_agent_in_thread(question: str, output: tk.Text, run_btn: ttk.Button):
    try:
        run_btn.config(state="disabled")
        # Ensure memory.json is created next to the exe
        os.chdir(os.path.dirname(resource_path("../memory.json")))

        print(f"Running LitReviewAgent for: {question}\n")
        run_agent(question)  # uses your existing pipeline
        print("\nDone.\n")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        run_btn.config(state="normal")


def main():
    root = tk.Tk()
    root.title("LitReviewAgent")
    root.geometry("900x600")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Enter your research question:").pack(anchor="w")

    question_var = tk.StringVar()
    entry = ttk.Entry(frm, textvariable=question_var)
    entry.pack(fill="x", pady=(4, 10))
    entry.focus()

    output = tk.Text(frm, wrap="word", height=25)
    output.pack(fill="both", expand=True)

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill="x", pady=(10, 0))

    def on_run():
        q = question_var.get().strip()
        if not q:
            messagebox.showwarning("Missing question", "Please enter a research question.")
            return
        t = threading.Thread(target=run_agent_in_thread, args=(q, output, run_btn), daemon=True)
        t.start()

    run_btn = ttk.Button(btn_row, text="Run Agent", command=on_run)
    run_btn.pack(side="left")

    def on_clear():
        output.delete("1.0", tk.END)

    ttk.Button(btn_row, text="Clear Output", command=on_clear).pack(side="left", padx=8)

    # Redirect stdout/stderr to the output box
    sys.stdout = TextRedirector(output)
    sys.stderr = TextRedirector(output)

    root.mainloop()


if __name__ == "__main__":
    main()
