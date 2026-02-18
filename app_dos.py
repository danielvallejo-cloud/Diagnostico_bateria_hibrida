# ============================================================
# SISTEMA DE DIAGNOSTICO DINAMICO DE BATERIA HIBRIDA
# INTERFAZ GRAFICA (Tkinter + Matplotlib)
# ============================================================

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# -------------------- MODULO ANALITICO ----------------------
# ============================================================

def analizar_bateria(file_path):

    df = pd.read_excel(file_path)

    module_columns = df.columns[1:21]
    modules = df[module_columns]

    current_column = [col for col in df.columns if "corriente" in col.lower()][0]
    current = df[current_column]

    time = np.arange(len(df))

    # Métricas dinámicas
    V_mean = modules.mean(axis=1)
    V_std = modules.std(axis=1)
    V_max = modules.max(axis=1)
    V_min = modules.min(axis=1)
    delta_V = V_max - V_min

    max_delta_V = delta_V.max()
    mean_delta_V = delta_V.mean()

    # Índice de desbalance
    imbalance_index = {
        col: np.sum(np.abs(modules[col] - V_mean))
        for col in module_columns
    }

    imbalance_df = pd.DataFrame.from_dict(
        imbalance_index, orient='index',
        columns=['Indice_Desbalance']
    ).sort_values(by='Indice_Desbalance', ascending=False)

    # Resistencia dinámica
    dI = np.diff(current)
    dynamic_resistance = {}

    for col in module_columns:
        dV = np.diff(modules[col])
        valid = np.abs(dI) > 0.1
        R = np.zeros_like(dV)
        R[valid] = dV[valid] / dI[valid]
        dynamic_resistance[col] = np.mean(np.abs(R[valid]))

    resistance_df = pd.DataFrame.from_dict(
        dynamic_resistance, orient='index',
        columns=['Resistencia_Dinamica']
    ).sort_values(by='Resistencia_Dinamica', ascending=False)

    # --- NUEVA LÓGICA DE DIAGNÓSTICO AGREGADA ---
    recomendaciones = []
    # Regla de Delta V (Estándar industria híbrida)
    if max_delta_V > 0.5:
        recomendaciones.append(f"🔴 CRÍTICO: Reemplazar el módulo {imbalance_df.index[0]}. Delta V ({max_delta_V:.2f}V) fuera de límite.")
    elif max_delta_V > 0.3:
        recomendaciones.append(f"🟠 ALERTA: Desbalance detectado en módulo {imbalance_df.index[0]}. Se sugiere mantenimiento/balanceo.")
    else:
        recomendaciones.append("🟢 SALUD: Los voltajes de los módulos están equilibrados.")

    # Regla de Resistencia / Ventilación
    if resistance_df['Resistencia_Dinamica'].iloc[0] > (resistance_df['Resistencia_Dinamica'].mean() * 1.4):
        recomendaciones.append("⚠️ VENTILACIÓN: Resistencia alta detectada. Limpiar ventilador y revisar ductos de enfriamiento.")
    
    diagnostico_final = "\n".join(recomendaciones)

    resultados = {
        "df": df,
        "modules": modules,
        "current": current,
        "time": time,
        "delta_V": delta_V,
        "max_delta_V": max_delta_V,
        "mean_delta_V": mean_delta_V,
        "imbalance_df": imbalance_df,
        "resistance_df": resistance_df,
        "diagnostico": diagnostico_final # Nuevo campo
    }

    return resultados


# ============================================================
# ---------------------- INTERFAZ GUI ------------------------
# ============================================================

class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Diagnóstico - Batería Híbrida - Daniel Vallejo")
        self.root.geometry("1200x750")

        self.file_path = None
        self.resultados = None

        self.crear_widgets()

    def crear_widgets(self):

        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=10)

        tk.Button(frame_top, text="Cargar Archivo Excel",
                  command=self.cargar_archivo).pack(side=tk.LEFT, padx=5)

        tk.Button(frame_top, text="Ejecutar Análisis",
                  command=self.ejecutar_analisis).pack(side=tk.LEFT, padx=5)

        # Métricas clave
        self.metricas_label = tk.Label(
            self.root,
            text="Métricas clave aparecerán aquí",
            font=("Arial", 12),
            justify="left"
        )
        self.metricas_label.pack(pady=10)

        # Notebook (tabs)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.tab_graficas = ttk.Frame(self.tabs)
        self.tab_diagnostico = ttk.Frame(self.tabs) # NUEVA PESTAÑA
        self.tab_desbalance = ttk.Frame(self.tabs)
        self.tab_resistencia = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_diagnostico, text="Diagnóstico Final") # Prioridad visual
        self.tabs.add(self.tab_graficas, text="Gráficas")
        self.tabs.add(self.tab_desbalance, text="Ranking Desbalance")
        self.tabs.add(self.tab_resistencia, text="Ranking Resistencia")

    def cargar_archivo(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx")]
        )
        if self.file_path:
            messagebox.showinfo("Archivo cargado", "Archivo seleccionado correctamente.")

    def ejecutar_analisis(self):

        if not self.file_path:
            messagebox.showerror("Error", "Debe cargar un archivo primero.")
            return

        try:
            self.resultados = analizar_bateria(self.file_path)
            self.mostrar_metricas()
            self.mostrar_graficas()
            self.mostrar_tablas()
            self.mostrar_diagnostico_final() # NUEVA FUNCIÓN
            messagebox.showinfo("Análisis completado", "El análisis finalizó correctamente.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def mostrar_metricas(self):

        r = self.resultados

        texto = f"""
Max ΔV: {r['max_delta_V']:.3f} V
Mean ΔV: {r['mean_delta_V']:.3f} V
Módulo Mayor Desbalance: {r['imbalance_df'].index[0]}
Módulo Mayor Resistencia: {r['resistance_df'].index[0]}
        """

        self.metricas_label.config(text=texto)

    def mostrar_diagnostico_final(self):
        # Limpieza de pestaña
        for widget in self.tab_diagnostico.winfo_children():
            widget.destroy()

        # Cuadro de texto para recomendaciones
        text_area = tk.Text(self.tab_diagnostico, font=("Verdana", 12), padx=20, pady=20, bg="#fdfdfd")
        text_area.pack(fill="both", expand=True)
        
        info = f"INFORME TÉCNICO DE RESULTADOS\n"
        info += "="*30 + "\n\n"
        info += self.resultados["diagnostico"] + "\n\n"
        info += f"DETALLE:\n"
        info += f"- Pico de desbalance en: {self.resultados['imbalance_df'].index[0]}\n"
        info += f"- Pico de resistencia en: {self.resultados['resistance_df'].index[0]}\n"
        
        text_area.insert(tk.END, info)
        text_area.config(state=tk.DISABLED)

    def mostrar_graficas(self):

        for widget in self.tab_graficas.winfo_children():
            widget.destroy()

        r = self.resultados

        fig, ax = plt.subplots(2, 2, figsize=(10, 7))

        # Voltajes módulos
        for col in r["modules"].columns:
            ax[0,0].plot(r["time"], r["modules"][col], alpha=0.3)
        ax[0,0].set_title("Voltajes Módulos")

        # Delta V
        ax[0,1].plot(r["time"], r["delta_V"])
        ax[0,1].set_title("Delta V")

        # Corriente
        ax[1,0].plot(r["time"], r["current"])
        ax[1,0].set_title("Corriente")

        # Delta V vs Corriente
        ax[1,1].scatter(r["current"], r["delta_V"], alpha=0.4)
        ax[1,1].set_title("Delta V vs Corriente")

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tab_graficas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def mostrar_tablas(self):

        for widget in self.tab_desbalance.winfo_children():
            widget.destroy()

        for widget in self.tab_resistencia.winfo_children():
            widget.destroy()

        # Desbalance
        tree1 = ttk.Treeview(self.tab_desbalance)
        tree1.pack(fill="both", expand=True)

        tree1["columns"] = ("Indice")
        tree1.heading("#0", text="Módulo")
        tree1.heading("Indice", text="Índice Desbalance")

        for idx, row in self.resultados["imbalance_df"].iterrows():
            tree1.insert("", "end", text=idx, values=(round(row[0],3)))

        # Resistencia
        tree2 = ttk.Treeview(self.tab_resistencia)
        tree2.pack(fill="both", expand=True)

        tree2["columns"] = ("Resistencia")
        tree2.heading("#0", text="Módulo")
        tree2.heading("Resistencia", text="Resistencia Dinámica")

        for idx, row in self.resultados["resistance_df"].iterrows():
            tree2.insert("", "end", text=idx, values=(round(row[0],6)))


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
