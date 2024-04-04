import customtkinter

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("dark-blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=12)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # menus to select source and target languages
        self.source_language_menu = customtkinter.CTkOptionMenu(self, values=["German"], command=self.change_source_language)
        self.source_language_menu.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.target_language_menu = customtkinter.CTkOptionMenu(self, values=["Englisch"], command=self.change_target_language)
        self.target_language_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # text area to input text
        self.source_text_area = customtkinter.CTkTextbox(self, wrap="word")
        self.source_text_area.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.target_text_area = customtkinter.CTkTextbox(self, wrap="word")
        self.target_text_area.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)


    def change_source_language(self, *args):
        print("Source language changed")

    def change_target_language(self, *args):
        print("Target language changed")

if __name__ == '__main__':
    source_languages = {
        "de": "German",
    }
    target_languages = {
        "en": "Englisch",
    }

    app = App()
    app.mainloop()
