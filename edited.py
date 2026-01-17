import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------- Theme constants (tuned to your screenshot style) ----------
BG = "#0B1020"          # deep navy
CARD = "#0F1A33"        # panel background
CARD2 = "#0C1730"       # slightly darker panel
BORDER = "#1E2A4A"      # subtle border
ACCENT = "#2F80ED"      # blue accent
TEXT = "#E8EEFF"        # near-white
MUTED = "#9DB0D0"       # muted text
GOOD = "#2ECC71"        # online
BAD = "#E74C3C"         # offline


class ProConsoleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("Riccati Console")
        self.geometry("1200x720")
        self.configure(fg_color=BG)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._topbar()
        self._sidebar()
        self._main_area()

    # ---------------- Top Bar ----------------
    def _topbar(self):
        bar = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 10))
        bar.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkLabel(
            bar,
            text="RiccatiQ",
            text_color=TEXT,
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        brand.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            bar,
            text="Numerical Control Utility",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(0, 2))

        # Right side: status + buttons
        right = ctk.CTkFrame(bar, fg_color=BG)
        right.grid(row=0, column=1, rowspan=2, sticky="e")

        self.status_pill = ctk.CTkLabel(
            right,
            text="● Online",
            text_color=TEXT,
            fg_color="#123025",
            corner_radius=999,
            padx=12,
            pady=6,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.status_pill.pack(side="right", padx=(10, 0))

        # Icon-like square buttons (no external icons needed)
        for t in ["⟳", "⚙", "⏻"]:
            b = ctk.CTkButton(
                right,
                text=t,
                width=42,
                height=34,
                corner_radius=10,
                fg_color=CARD,
                hover_color="#132244",
                border_color=BORDER,
                border_width=1,
                text_color=TEXT,
                command=lambda x=t: self._top_action(x),
            )
            b.pack(side="right", padx=6)

    def _top_action(self, x):
        if x == "⏻":
            # toggle demo status
            if "Online" in self.status_pill.cget("text"):
                self.status_pill.configure(text="● Offline", fg_color="#2A1212")
            else:
                self.status_pill.configure(text="● Online", fg_color="#123025")

    # ---------------- Sidebar ----------------
    def _sidebar(self):
        self.side = ctk.CTkFrame(
            self, fg_color=CARD, corner_radius=18, border_color=BORDER, border_width=1
        )
        self.side.grid(row=1, column=0, sticky="nsw", padx=(16, 10), pady=(0, 16))
        self.side.grid_rowconfigure(99, weight=1)

        title = ctk.CTkLabel(self.side, text="Configuration", text_color=TEXT,
                             font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(anchor="w", padx=14, pady=(14, 8))

        # Small section label
        self._section_label(self.side, "Time")
        self.t0 = self._entry_row(self.side, "t0", "0")
        self.tf = self._entry_row(self.side, "tf", "10")

        self._section_label(self.side, "Initial condition")
        self.y0 = self._entry_row(self.side, "y(t0)", "0.1")

        self._section_label(self.side, "Parameters")
        self.k = self._entry_row(self.side, "k", "1.0")
        self.m = self._entry_row(self.side, "m", "0.5")
        self.alpha = self._entry_row(self.side, "alpha", "0.0")

        self._section_label(self.side, "Actions")
        solve = ctk.CTkButton(
            self.side,
            text="Solve",
            height=40,
            corner_radius=14,
            fg_color=ACCENT,
            hover_color="#2566C7",
            text_color="white",
            command=self.on_solve,
        )
        solve.pack(fill="x", padx=14, pady=(10, 8))

        clear = ctk.CTkButton(
            self.side,
            text="Clear",
            height=40,
            corner_radius=14,
            fg_color=CARD2,
            hover_color="#122246",
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            command=self.on_clear,
        )
        clear.pack(fill="x", padx=14, pady=(0, 14))

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color=MUTED,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))

    def _entry_row(self, parent, label, default):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(row, text=label, text_color=TEXT, width=80, anchor="w").pack(side="left")

        ent = ctk.CTkEntry(
            row,
            fg_color=CARD2,
            border_color=BORDER,
            border_width=1,
            text_color=TEXT,
            corner_radius=12,
            height=34,
        )
        ent.insert(0, default)
        ent.pack(side="right", fill="x", expand=True)
        return ent

    # ---------------- Main Area ----------------
    def _main_area(self):
        self.main = ctk.CTkFrame(
            self, fg_color=CARD, corner_radius=18, border_color=BORDER, border_width=1
        )
        self.main.grid(row=1, column=1, sticky="nsew", padx=(10, 16), pady=(0, 16))
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Tabs row (fake tabs styled like your screenshot)
        tabs = ctk.CTkFrame(self.main, fg_color="transparent")
        tabs.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        for i, name in enumerate(["Plot", "Diagnostics", "Console"]):
            btn = ctk.CTkButton(
                tabs,
                text=name,
                height=34,
                corner_radius=12,
                fg_color=("#132244" if i == 0 else CARD2),
                hover_color="#132244",
                border_color=BORDER,
                border_width=1,
                text_color=TEXT,
            )
            btn.pack(side="left", padx=6)

        # Plot card
        plot_card = ctk.CTkFrame(self.main, fg_color=CARD2, corner_radius=16,
                                 border_color=BORDER, border_width=1)
        plot_card.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        plot_card.grid_rowconfigure(0, weight=1)
        plot_card.grid_columnconfigure(0, weight=1)

        # Matplotlib
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("y(t)")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("y")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    # ---------------- Actions ----------------
    def on_clear(self):
        self.ax.clear()
        self.ax.set_title("y(t)")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("y")
        self.canvas.draw()

    def on_solve(self):
        # Demo solver (replace with your Riccati numeric solver)
        t0 = float(self.t0.get())
        tf = float(self.tf.get())
        y0 = float(self.y0.get())

        k = float(self.k.get())
        m = float(self.m.get())
        alpha = float(self.alpha.get())

        t = np.linspace(t0, tf, 800)
        # demo curve (replace)
        y = y0 + 0.1*np.sin(t) - k*0.02*t + m*0.01*(t**2) + alpha*0.05

        self.ax.plot(t, y, label="solution")
        self.ax.legend(loc="best")
        self.canvas.draw()


if __name__ == "__main__":
    app = ProConsoleApp()
    app.mainloop()