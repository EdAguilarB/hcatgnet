import tkinter as tk
from tkinter import ttk
from options.base_options import BaseOptions
import os

# Experiment scripts
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import denoise_graphs, GNNExplainer_node_feats, shapley_analysis

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Run All Experiments")
        self.geometry("500x400")
        self.configure(bg='#f0f0f0')

        # Create a style
        self.style = ttk.Style(self)
        self.style.configure('TCheckbutton', font=('Helvetica', 14))
        self.style.configure('TButton', font=('Helvetica', 14), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 16), background='#f0f0f0')

        # Create the variables
        self.train_GNN_var = tk.BooleanVar(value=False)
        self.train_tml_var = tk.BooleanVar(value=False)
        self.predict_unseen_var = tk.BooleanVar(value=False)
        self.compare_models_var = tk.BooleanVar(value=False)
        self.denoise_graph_var = tk.BooleanVar(value=False)
        self.GNNExplainer_var = tk.BooleanVar(value=False)
        self.shapley_analysis_var = tk.BooleanVar(value=False)

        self.denoise_reactions = None  # To store the denoise reactions value
        self.denoise_mol = None  # To store the denoise molecule value
        self.denoise_based_on = None  # To store the denoise based on value
        self.norm = None  # To store the normalize value


        self.create_widgets()

    def create_widgets(self):
        # Title label
        title_label = ttk.Label(self, text="Run Experiments", style='TLabel')
        title_label.pack(pady=10)

        # Description label
        description_label = ttk.Label(self, text="Select the operations to run:", style='TLabel')
        description_label.pack(pady=5)

        # Checkbuttons
        options_frame = ttk.Frame(self)
        options_frame.pack(pady=10)

        ttk.Checkbutton(options_frame, text="Train GNN", variable=self.train_GNN_var).grid(column=0, row=0, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Train TML Model", variable=self.train_tml_var).grid(column=0, row=1, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Predict Unseen Data", variable=self.predict_unseen_var).grid(column=0, row=2, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Compare Models", variable=self.compare_models_var).grid(column=0, row=3, sticky=tk.W, padx=10, pady=5)

        denoise_checkbutton = ttk.Checkbutton(options_frame, text="Denoise Graph", variable=self.denoise_graph_var, command=self.open_denoise_window)
        denoise_checkbutton.grid(column=0, row=4, sticky=tk.W, padx=10, pady=5)

        ttk.Checkbutton(options_frame, text="Run GNNExplainer", variable=self.GNNExplainer_var).grid(column=0, row=5, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Shapley Analysis", variable=self.shapley_analysis_var).grid(column=0, row=6, sticky=tk.W, padx=10, pady=5)

        # Run button
        run_button = ttk.Button(self, text="Run", command=self.run_experiments)
        run_button.pack(pady=20)

    def open_denoise_window(self):
        if self.denoise_graph_var.get():
            denoise_window = tk.Toplevel(self)
            denoise_window.title("Denoise Graph Options")
            denoise_window.geometry("400x300")
            denoise_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_denoise_molecule = ttk.Label(denoise_window, text="Denoise molecule:", style='TLabel')
            label_denoise_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.denoise_molecule_var = tk.StringVar()

            options = ['ligand', 'substrate', 'boron']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.denoise_based_on_var = tk.StringVar()

            options = ['All', 'atom_identity', 'degree', 'hyb', 'aromatic', 'ring', 'chiral', 'conf']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_based_on_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_denoise_reaction = ttk.Label(denoise_window, text="Denoise reaction number:", style='TLabel')
            label_denoise_reaction.pack()

            self.denoise_reaction_entry = ttk.Entry(denoise_window, font=('Helvetica', 14))
            self.denoise_reaction_entry.pack(pady=5)
    
            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)


            # Checkbox for True/False
            self.checkbox_norm = tk.BooleanVar()
            self.checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(denoise_window, text="Normalize attribution scores", variable=self.checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(denoise_window, text="Apply", command=lambda: self.apply_denoise_options(denoise_window))
            apply_button.pack(pady=10)


    def apply_denoise_options(self, window):
        try:
            self.denoise_reactions = int(self.denoise_reaction_entry.get())
            self.denoise_mol = self.denoise_molecule_var.get()
            self.denoise_based_on = self.denoise_based_on_var.get()
            self.norm = self.checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()

    def run_experiments(self):
        opt = BaseOptions().parse()


        # Update opt with user-provided denoise_reactions value
        if self.denoise_graph_var.get() and self.denoise_reactions is not None:
            opt.denoise_reactions = self.denoise_reactions
            opt.denoise_mol = self.denoise_mol
            opt.denoise_based_on = self.denoise_based_on
            opt.norm = self.norm
        

        if self.train_GNN_var.get():
            train_network_nested_cv()

        if self.train_tml_var.get():
            train_tml_model_nested_cv()

        if self.predict_unseen_var.get():
            predict_final_test()

        if self.compare_models_var.get():
            plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4]))
            plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, 'final_test'))

        if self.denoise_graph_var.get():
            if self.denoise_reactions is not None:
                denoise_graphs(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))
            else:
                print("Denoise reactions value not provided")

        if self.GNNExplainer_var.get():
            GNNExplainer_node_feats(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))

        if self.shapley_analysis_var.get():
            shapley_analysis(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


if __name__ == "__main__":
    app = Application()
    app.mainloop()






