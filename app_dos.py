# ============================================================
# SISTEMA DE DIAGNOSTICO DINAMICO DE BATERIA HIBRIDA
# INTERFAZ STREAMLIT - VERSION MEJORADA SIN DEPENDENCIAS EXTERNAS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# -------------------- MODULO ANALITICO ----------------------
# ============================================================

def analizar_bateria(file):
    df = pd.read_excel(file)
    
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
    imbalance_index = {col: np.sum(np.abs(modules[col] - V_mean)) for col in module_columns}
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

    # Diagnóstico basado en reglas
    recomendaciones = []
    if max_delta_V > 0.5:
        recomendaciones.append(f"🔴 CRÍTICO: Reemplazar el módulo {imbalance_df.index[0]}. Delta V ({max_delta_V:.2f}V) fuera de límite.")
    elif max_delta_V > 0.3:
        recomendaciones.append(f"🟠 ALERTA: Desbalance detectado en módulo {imbalance_df.index[0]}. Se sugiere mantenimiento/balanceo.")
    else:
        recomendaciones.append("🟢 SALUD: Los voltajes de los módulos están equilibrados.")

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
        "diagnostico": diagnostico_final
    }

    return resultados

# ============================================================
# ---------------------- INTERFAZ STREAMLIT -----------------
# ============================================================

st.set_page_config(page_title="Diagnóstico Batería Híbrida", layout="wide")
st.title("💡 Sistema de Diagnóstico Dinámico - Batería Híbrida")
st.markdown("**Autor:** Daniel Vallejo")

# Subida de archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type="xlsx")

if uploaded_file:
    resultados = analizar_bateria(uploaded_file)

    # --- PESTAÑAS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Diagnóstico Final", "Gráficas", "Ranking Desbalance", "Ranking Resistencia"])

    # ---------------- TAB 1: Diagnóstico Final ----------------
    with tab1:
        st.subheader("Informe Técnico de Resultados")
        st.text_area("Detalle de diagnóstico", resultados['diagnostico'], height=200)
        
        # Panel de métricas con colores
        col1, col2, col3 = st.columns(3)
        col1.metric("ΔV Máximo", f"{resultados['max_delta_V']:.3f} V", delta=f"{resultados['mean_delta_V']:.3f} V")
        col2.metric("Módulo Pico Desbalance", resultados['imbalance_df'].index[0])
        col3.metric("Módulo Pico Resistencia", resultados['resistance_df'].index[0])

    # ---------------- TAB 2: Gráficas ----------------
    with tab2:
        st.subheader("Visualización de Gráficas")
        
        # Filtro por módulos
        modulos_seleccionados = st.multiselect(
            "Selecciona módulos a graficar",
            resultados["modules"].columns,
            default=resultados["modules"].columns[:5]
        )

        if modulos_seleccionados:
            fig, ax = plt.subplots(2, 2, figsize=(12, 7))

            # Voltajes módulos
            for col in modulos_seleccionados:
                ax[0,0].plot(resultados["time"], resultados["modules"][col], alpha=0.3)
            ax[0,0].set_title("Voltajes Módulos")

            # Delta V
            ax[0,1].plot(resultados["time"], resultados["delta_V"], color='orange')
            ax[0,1].set_title("Delta V")

            # Corriente
            ax[1,0].plot(resultados["time"], resultados["current"], color='green')
            ax[1,0].set_title("Corriente")

            # Delta V vs Corriente
            ax[1,1].scatter(resultados["current"], resultados["delta_V"], alpha=0.4, color='red')
            ax[1,1].set_title("Delta V vs Corriente")

            fig.tight_layout()
            st.pyplot(fig)

    # ---------------- TAB 3: Ranking Desbalance ----------------
    with tab3:
        st.subheader("Ranking de Desbalance por Módulo")
        st.dataframe(resultados['imbalance_df'].style.background_gradient(cmap='Reds'))

    # ---------------- TAB 4: Ranking Resistencia ----------------
    with tab4:
        st.subheader("Ranking de Resistencia Dinámica por Módulo")
        st.dataframe(resultados['resistance_df'].style.background_gradient(cmap='Reds'))

    # ---------------- BOTON DE EXPORTACION ----------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        resultados['df'].to_excel(writer, sheet_name='Datos Originales', index=False)
        resultados['imbalance_df'].to_excel(writer, sheet_name='Ranking Desbalance')
        resultados['resistance_df'].to_excel(writer, sheet_name='Ranking Resistencia')
    output.seek(0)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=output,
        file_name="diagnostico_bateria.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
