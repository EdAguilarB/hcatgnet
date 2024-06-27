import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as fd
from options.base_options import BaseOptions
import os

# Experiment scripts
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import denoise_graphs, GNNExplainer_node_feats, shapley_analysis

from icecream import ic

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Run All Experiments")
        self.geometry("500x400")
        self.configure(bg='#f0f0f0')

        # Create a style
        self.style = ttk.Style(self)
        self.style.configure('TCheckbutton', font=('Avenir Next', 14))
        self.style.configure('TButton', font=('Avenir Next', 14), padding=10)
        self.style.configure('TLabel', font=('Avenir Next', 16), background='#f0f0f0')

        # Create the variables
        self.train_GNN_var = tk.BooleanVar(value=False)
        self.train_tml_var = tk.BooleanVar(value=False)
        self.predict_unseen_var = tk.BooleanVar(value=False)
        self.compare_models_var = tk.BooleanVar(value=False)
        self.denoise_graph_var = tk.BooleanVar(value=False)
        self.GNNExplainer_var = tk.BooleanVar(value=False)
        self.shapley_analysis_var = tk.BooleanVar(value=False)

        # train_GNN variables
        self.data_entry = None
        self.log_dir_entry = None
        self.log_dir_name_entry = None

        # train_tml variables
        self.data_entry_tml = None
        self.log_dir_entry_tml = None
        self.log_dir_name_entry_tml = None
        self.train_algorithm_tml = None

        # Denoise variables
        self.denoise_reactions = None  # To store the denoise reactions value
        self.denoise_mol = None  # To store the denoise molecule value
        self.denoise_based_on = None  # To store the denoise based on value
        self.denoise_outer = None  # To store the outer value
        self.denoise_inner = None # To store the inner value
        self.denoise_norm = None  # To store the normalize value

        # GNNExplainer variables
        self.explain_reactions = None
        self.explain_mol = None
        self.explain_based_on = None
        self.explain_outer = None
        self.explain_inner = None
        self.explain_norm = None

        # Shap variables
        self.shap_reactions = None  # To store the shap reactions value
        self.shap_mol = None
        self.shap_based_on = None
        self.shap_outer = None
        self.shap_inner = None
        self.shap_norm = None

        self.tml_algorithm_dict = {'Random Forest': 'rf',
                                   'Gradient Boosting': 'gb',
                                   'Linear Regression': 'lr'}
        

        self.denoise_mol_dict = {'Ligand': 'ligand',
                                    'Substrate': 'substrate',
                                    'Organoboron Reagent': 'boron',
                                    'All': None}
        
        self.denoise_based_on_dict = {'All': None,
                                      'Atom Identity': 'atom_identity',
                                        'Degree': 'degree',
                                        'Hybridization': 'hyb',
                                        'Aromaticity': 'aromatic',
                                        'Atom in Ring': 'ring',
                                        'Chirality': 'chiral',
                                        'Ligand Configuration': 'conf'}


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

        train_GNN_checkbutton = ttk.Checkbutton(options_frame, text="Train GNN", variable=self.train_GNN_var, command=self.open_train_GNN_window)
        train_GNN_checkbutton.grid(column=0, row=0, sticky=tk.W, padx=10, pady=5)

        train_TML_checkbutton = ttk.Checkbutton(options_frame, text="Train TML Model", variable=self.train_tml_var, command=self.open_train_tml_window)
        train_TML_checkbutton.grid(column=0, row=1, sticky=tk.W, padx=10, pady=5)

        ttk.Checkbutton(options_frame, text="Predict Unseen Data", variable=self.predict_unseen_var).grid(column=0, row=2, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Compare Models", variable=self.compare_models_var).grid(column=0, row=3, sticky=tk.W, padx=10, pady=5)

        denoise_checkbutton = ttk.Checkbutton(options_frame, text="Denoise Graph", variable=self.denoise_graph_var, command=self.open_denoise_window)
        denoise_checkbutton.grid(column=0, row=4, sticky=tk.W, padx=10, pady=5)

        GNNExp_checkbutton = ttk.Checkbutton(options_frame, text="Run GNNExplainer", variable=self.GNNExplainer_var, command=self.open_GNNEx_window)
        GNNExp_checkbutton.grid(column=0, row=5, sticky=tk.W, padx=10, pady=5)

        shap_checkbutton = ttk.Checkbutton(options_frame, text="Shapley Analysis", variable=self.shapley_analysis_var, command=self.open_shap_window)
        shap_checkbutton.grid(column=0, row=6, sticky=tk.W, padx=10, pady=5)

        # Run button
        run_button = ttk.Button(self, text="Run", command=self.run_experiments)
        run_button.pack(pady=20)


    def browse_file(self):
        file_path = fd.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, file_path)

    def browse_directory(self):
        dir_path = fd.askdirectory()
        if dir_path:
            self.log_dir_entry.delete(0, tk.END)
            self.log_dir_entry.insert(0, dir_path)

    def open_train_GNN_window(self):
        if self.train_GNN_var.get():
            # Create a new window
            train_GNN_window = tk.Toplevel(self)
            train_GNN_window.title("Train GNN Options")
            train_GNN_window.geometry("600x650")
            train_GNN_window.configure(bg='#f0f0f0')

            # Select csv file with the data
            label_data = ttk.Label(train_GNN_window, text="Select the csv file with the data:", style='TLabel')
            label_data.pack(pady=10)

            self.data_entry = ttk.Entry(train_GNN_window, font=('Helvetica', 14), width=50)
            self.data_entry.pack(pady=5)

            browse_button = ttk.Button(train_GNN_window, text="Browse", command=self.browse_file)
            browse_button.pack(pady=5)

            # Log directory
            log_dir_label = ttk.Label(train_GNN_window, text="Log directory:", style='TLabel')
            log_dir_label.pack(pady=10)

            self.log_dir_entry = ttk.Entry(train_GNN_window, font=('Helvetica', 14), width=50)
            self.log_dir_entry.pack(pady=5)

            log_dir_button = ttk.Button(train_GNN_window, text="Browse", command=self.browse_directory)
            log_dir_button.pack(pady=5)

            # Log directory name
            log_dir_name_label = ttk.Label(train_GNN_window, text="Log directory name:", style='TLabel')
            log_dir_name_label.pack(pady=10)

            self.log_dir_name_entry = ttk.Entry(train_GNN_window, font=('Helvetica', 14), width=50)
            self.log_dir_name_entry.pack(pady=5)

            # Apply button
            apply_button = ttk.Button(train_GNN_window, text="Apply", command=lambda: self.apply_GNN_train_options(train_GNN_window))
            apply_button.pack(pady=10)

    def apply_GNN_train_options(self, window):
        try:
            self.data_entry = self.data_entry.get()
            self.log_dir_entry = self.log_dir_entry.get()
            self.log_dir_name_entry = self.log_dir_name_entry.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_train_tml_window(self):
        if self.train_tml_var.get():
            # Create a new window
            train_tml_window = tk.Toplevel(self)
            train_tml_window.title("Train TML Options")
            train_tml_window.geometry("600x650")
            train_tml_window.configure(bg='#f0f0f0')

            # Select csv file with the data
            label_data = ttk.Label(train_tml_window, text="Select the csv file with the data:", style='TLabel')
            label_data.pack(pady=10)

            self.data_entry = ttk.Entry(train_tml_window, font=('Helvetica', 14), width=50)
            self.data_entry.pack(pady=5)

            browse_button = ttk.Button(train_tml_window, text="Browse", command=self.browse_file)
            browse_button.pack(pady=5)

            # Log directory
            log_dir_label = ttk.Label(train_tml_window, text="Log directory:", style='TLabel')
            log_dir_label.pack(pady=10)

            self.log_dir_entry = ttk.Entry(train_tml_window, font=('Helvetica', 14), width=50)
            self.log_dir_entry.pack(pady=5)

            log_dir_button = ttk.Button(train_tml_window, text="Browse", command=self.browse_directory)
            log_dir_button.pack(pady=5)

            # Log directory name
            log_dir_name_label = ttk.Label(train_tml_window, text="Log directory name:", style='TLabel')
            log_dir_name_label.pack(pady=10)

            self.log_dir_name_entry = ttk.Entry(train_tml_window, font=('Helvetica', 14), width=50)
            self.log_dir_name_entry.pack(pady=5)

            # Denoise molecule label
            label_choose_tml = ttk.Label(train_tml_window, text="Choose the tml algorithm to train:", style='TLabel')
            label_choose_tml.pack(pady=10)

            # Radio button options for denoise molecule
            self.train_tml_algorithm = tk.StringVar()

            options = ['Gradient Boosting', 'Random Forest', 'Linear Regression']

            for option in options:
                rb = ttk.Radiobutton(train_tml_window, text=option, variable=self.train_tml_algorithm, value=option)
                rb.pack(anchor='center', padx=20)

            # Apply button
            apply_button = ttk.Button(train_tml_window, text="Apply", command=lambda: self.apply_tml_train_options(train_tml_window))
            apply_button.pack(pady=10)

    def apply_tml_train_options(self, window):
        try:
            self.data_entry_tml = self.data_entry.get()
            self.log_dir_entry_tml = self.log_dir_entry.get()
            self.log_dir_name_entry_tml = self.log_dir_name_entry.get()
            self.train_algorithm_tml = self.tml_algorithm_dict[self.train_tml_algorithm.get()]

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_denoise_window(self):

        if self.denoise_graph_var.get():

            # Create a new window
            denoise_window = tk.Toplevel(self)
            denoise_window.title("Denoise Graph Options")
            denoise_window.geometry("400x650")
            denoise_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_denoise_molecule = ttk.Label(denoise_window, text="Denoise molecule:", style='TLabel')
            label_denoise_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.denoise_molecule_var = tk.StringVar()

            options = ['Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.denoise_based_on_var = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_based_on_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_denoise_reaction = ttk.Label(denoise_window, text="Denoise reaction number:", style='TLabel')
            label_denoise_reaction.pack()

            self.denoise_reaction_entry = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_reaction_entry.pack(pady=5)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(denoise_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.denoise_outer = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_outer.pack(pady=5)

            label_explain_inner = ttk.Label(denoise_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.denoise_inner = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_inner.pack(pady=5)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)


            # Checkbox for True/False
            self.den_checkbox_norm = tk.BooleanVar()
            self.den_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(denoise_window, text="Normalize attribution scores", variable=self.den_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(denoise_window, text="Apply", command=lambda: self.apply_denoise_options(denoise_window))
            apply_button.pack(pady=10)

    def apply_denoise_options(self, window):
        try:
            self.denoise_reactions = int(self.denoise_reaction_entry.get())
            self.denoise_mol = self.denoise_mol_dict[self.denoise_molecule_var.get()]
            self.denoise_based_on = self.denoise_based_on_dict[self.denoise_based_on_var.get()]
            self.outer = int(self.denoise_outer.get())
            self.inner = int(self.denoise_inner.get())
            self.norm = self.den_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_GNNEx_window(self):
        if self.GNNExplainer_var.get():
            gnnex_window = tk.Toplevel(self)
            gnnex_window.title("GNNExplainer Options")
            gnnex_window.geometry("400x650")
            gnnex_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_explain_molecule = ttk.Label(gnnex_window, text="Analyze Molecule:", style='TLabel')
            label_explain_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.explain_molecule_var = tk.StringVar()

            options = ['All', 'Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(gnnex_window, text=option, variable=self.explain_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.explain_feature = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(gnnex_window, text=option, variable=self.explain_feature, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_explain_reaction = ttk.Label(gnnex_window, text="Analyze reaction number:", style='TLabel')
            label_explain_reaction.pack()

            self.explain_reaction_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_reaction_entry.pack(pady=5)


            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(gnnex_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.explain_outer_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_outer_entry.pack(pady=5)

            label_explain_inner = ttk.Label(gnnex_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.explain_inner_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_inner_entry.pack(pady=5)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)


            # Checkbox for True/False
            self.exp_checkbox_norm = tk.BooleanVar()
            self.exp_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(gnnex_window, text="Normalize attribution scores", variable=self.exp_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(gnnex_window, text="Apply", command=lambda: self.apply_GNNExp_options(gnnex_window))
            apply_button.pack(pady=10)


    def apply_GNNExp_options(self, window):
        try:
            self.explain_reactions = int(self.explain_reaction_entry.get())
            self.explain_mol = self.denoise_mol_dict[self.explain_molecule_var.get()]
            self.explain_based_on = self.denoise_based_on_dict[self.explain_feature.get()]
            self.explain_outer = int(self.explain_outer_entry.get())
            self.explain_inner = int(self.explain_inner_entry.get())
            self.explain_norm = self.exp_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()

    def open_shap_window(self):
        if self.shapley_analysis_var.get():
            shap_window = tk.Toplevel(self)
            shap_window.title("Shap Value Sampling Options")
            shap_window.geometry("400x650")
            shap_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_denoise_molecule = ttk.Label(shap_window, text="Analyze Molecule:", style='TLabel')
            label_denoise_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.shap_molecule_var = tk.StringVar()

            options = ['Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(shap_window, text=option, variable=self.shap_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.shap_based_on_var = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(shap_window, text=option, variable=self.shap_based_on_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_denoise_reaction = ttk.Label(shap_window, text="Analyze reaction number:", style='TLabel')
            label_denoise_reaction.pack()

            self.shap_reaction_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_reaction_entry.pack(pady=5)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(shap_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.shap_outer_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_outer_entry.pack(pady=5)

            label_explain_inner = ttk.Label(shap_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.shap_inner_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_inner_entry.pack(pady=5)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Checkbox for True/False
            self.shap_checkbox_norm = tk.BooleanVar()
            self.shap_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(shap_window, text="Normalize attribution scores", variable=self.shap_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(shap_window, text="Apply", command=lambda: self.apply_shap_options(shap_window))
            apply_button.pack(pady=10)


    def apply_shap_options(self, window):
        try:
            self.shap_reactions = int(self.shap_reaction_entry.get())
            self.shap_mol = self.denoise_mol_dict[self.shap_molecule_var.get()]
            self.shap_based_on = self.denoise_based_on_dict[self.shap_based_on_var.get()]
            self.shap_outer = int(self.shap_outer_entry.get())
            self.shap_inner = int(self.shap_inner_entry.get())
            self.shap_norm = self.shap_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()

    def run_experiments(self):
        opt = BaseOptions().parse()
        

        if self.train_GNN_var.get():
            path, filename = os.path.split(self.data_entry)
            opt.root = os.path.dirname(path)
            opt.filename = filename
            opt.log_dir_results = self.log_dir_name_entry
            train_network_nested_cv(opt)

        if self.train_tml_var.get():
            path, filename = os.path.split(self.data_entry_tml)
            opt.root = os.path.dirname(path)
            opt.filename = filename
            opt.log_dir_results = os.path.join(self.log_dir_entry_tml, self.log_dir_name_entry_tml)
            opt.tml_algorithm = self.train_algorithm_tml
            train_tml_model_nested_cv(opt)

        if self.predict_unseen_var.get():
            predict_final_test()

        if self.compare_models_var.get():
            plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4]))
            plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, 'final_test'))

        
        # Update opt with user-provided denoise_reactions value
        if self.denoise_graph_var.get():
            opt.denoise_reactions = self.denoise_reactions
            opt.denoise_mol = self.denoise_mol
            opt.denoise_based_on = self.denoise_based_on
            opt.explain_model = [self.outer, self.inner]
            opt.norm = self.norm

            # Run denoise_graphs
            denoise_graphs(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


        if self.GNNExplainer_var.get():
            # Update opt with user-provided explain_reactions value
            opt.explain_model = [self.explain_outer, self.explain_inner]
            # Run GNNExplainer_node_feats
            GNNExplainer_node_feats(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


        if self.shapley_analysis_var.get():
            # Update opt with user-provided shap_reactions value
            opt.shap_index = self.shap_reactions
            opt.explain_model = [self.shap_outer, self.shap_inner]
            # Run shapley_analysis
            shapley_analysis(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


if __name__ == "__main__":
    app = Application()
    app.mainloop()






