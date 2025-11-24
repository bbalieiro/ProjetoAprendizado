import streamlit as st
import pandas as pd
from modelo import TreinadorModelo

st.title("Treino e Teste - Regressão Linear")

# Carregar modelo existente ou criar novo
modelo = TreinadorModelo()

st.header("Treinar modelo")
file_train = st.file_uploader("Upload CSV de treino", type=["csv"], key="train")

if file_train:
    df = pd.read_csv(file_train)
    metrica = modelo.treinar(df)
    st.success(f"Treino concluído! Desempenho: {metrica}")
    modelo.salvar()

st.header("Testar modelo")
file_test = st.file_uploader("Upload CSV de teste", type=["csv"], key="test")

tem_rotulos = st.checkbox("O CSV possui rótulos? (coluna 'time')")

if file_test:
    df = pd.read_csv(file_test)
    preds, desempenho = modelo.testar(df, tem_rotulos)

    st.download_button(
        "Baixar previsões",
        preds.to_csv(index=False),
        file_name="predicoes.csv"
    )

    if desempenho is not None:
        st.write("Desempenho real:", desempenho)

if st.button("Resetar modelo"):
    modelo.resetar()
    st.success("Modelo resetado!")
