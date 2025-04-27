import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binom, poisson, norm

# Criação de abas para diferentes distribuições
tab1, tab2, tab3, tab4 = st.tabs([
    "Distribuição Binomial", 
    "Distribuição Poisson", 
    "Distribuição Normal", 
    "Simulação de ROI"
])

# --- QUESTÃO 1 ---
with tab1:
    st.header("Distribuição Binomial: Overbooking Aéreo")
    st.markdown("""
    ### Questão 1 - Análise de Overbooking na Companhia Aérea Aérea Confiável

    **Contexto:** Uma companhia aérea vendeu **130 passagens** para um voo com **capacidade para 120 passageiros**. A expectativa é que nem todos compareçam, considerando uma **probabilidade de comparecimento de 88%**.

    O objetivo é analisar a **probabilidade de overbooking**, isto é, a chance de mais de 120 passageiros comparecerem.
    """)

    vendidos = 130
    p = 0.88
    capacidade = 120

    prob_overbooking = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("Probabilidade de Overbooking (>120 passageiros)", f"{prob_overbooking*100:.2f}%")

    st.markdown("---")
    st.subheader("Limite de Risco de Overbooking (até 7%)")
    st.markdown("Analisamos a quantidade máxima de passagens que poderiam ser vendidas mantendo o risco de overbooking inferior a 7%.")

    limite_passagens = np.arange(120, 151)
    riscos = [1 - binom.cdf(capacidade, n, p) for n in limite_passagens]
    df_riscos = pd.DataFrame({"Passagens Vendidas": limite_passagens, "Risco de Overbooking": riscos})

    fig_risco = px.line(df_riscos, x="Passagens Vendidas", y="Risco de Overbooking",
                        title="Risco de Overbooking conforme Número de Passagens Vendidas",
                        labels={"Risco de Overbooking": "Probabilidade"})
    fig_risco.add_hline(y=0.07, line_dash="dash", line_color="red")
    st.plotly_chart(fig_risco, use_container_width=True)

    max_seguro = df_riscos[df_riscos["Risco de Overbooking"] <= 0.07]["Passagens Vendidas"].max()
    if not np.isnan(max_seguro):
        st.success(f"Até {max_seguro} passagens podem ser vendidas sem ultrapassar 7% de risco de overbooking.")
    else:
        st.error("Nenhuma quantidade de passagens vendidas mantém o risco abaixo de 7%.")

    st.markdown("---")
    st.subheader("Análise de Viabilidade Financeira: Vender +10 Assentos")

    st.markdown(f"""
    Considerando a venda de **10 passagens a mais** (130 no total), o risco de overbooking seria de aproximadamente
    **{prob_overbooking*100:.2f}%**. 

    Se o custo de realocar passageiros for de **R$ 500,00 por cliente excedente**, avaliamos o custo esperado e o lucro potencial.
    """)

    custo_indenizacao = 500
    custo_esperado = prob_overbooking * custo_indenizacao * (vendidos - capacidade)
    lucro_extra = 10 * 500

    st.metric("Lucro Bruto com 10 Passagens Extras", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado com Overbooking", f"R$ {custo_esperado:,.2f}".replace(",", "."))

    if lucro_extra > custo_esperado:
        st.success("A venda de 10 passagens extras é **financeiramente vantajosa**.")
    else:
        st.warning("A venda extra apresenta risco financeiro elevado. Avaliação cautelosa recomendada.")

# --- DISTRIBUIÇÃO POISSON ---
with tab2:
    st.header("Distribuição de Poisson - Chegada de Clientes por Hora")
    st.markdown("""
    Modelagem da chegada de clientes a uma loja ao longo do dia, assumindo uma distribuição de Poisson.
    """)

    lambda_val = st.slider("Taxa média de chegada de clientes (por hora)", 1, 20, 5)
    horas = np.arange(0, 15)
    probabilidades = poisson.pmf(horas, mu=lambda_val)
    df_poisson = pd.DataFrame({"Número de Clientes": horas, "Probabilidade": probabilidades})

    fig_poisson = px.bar(df_poisson, x="Número de Clientes", y="Probabilidade",
                         title="Distribuição de Poisson - Número de Clientes por Hora")
    st.plotly_chart(fig_poisson, use_container_width=True)

# --- DISTRIBUIÇÃO NORMAL ---
with tab3:
    st.header("Distribuição Normal - Vendas de Produtos")
    st.markdown("""
    Simulamos a distribuição normal do número de produtos vendidos em determinado período.
    Ajuste os parâmetros para observar diferentes cenários.
    """)

    media = st.slider("Média de Vendas", 50, 150, 100)
    desvio = st.slider("Desvio Padrão", 5, 30, 15)

    x = np.linspace(media - 4*desvio, media + 4*desvio, 200)
    y = norm.pdf(x, media, desvio)
    df_normal = pd.DataFrame({"Quantidade Vendida": x, "Densidade": y})

    fig_normal = px.area(df_normal, x="Quantidade Vendida", y="Densidade",
                         title="Distribuição Normal das Vendas", labels={"Densidade": "Função Densidade"})
    st.plotly_chart(fig_normal, use_container_width=True)

# --- QUESTÃO 2 ---
with tab4:
    st.header("Simulação Interativa de ROI - Novo Sistema de Informação")
    st.markdown("""
    ### Questão 2 - ROI de Sistema de Informação

    **Contexto:** Avaliação de um novo sistema de informação, com investimento inicial de R$ 50.000, expectativa de receita de R$ 80.000, e custo operacional anual de R$ 10.000.
    """)

    receita_esperada = 80000
    custo_operacional = 10000
    investimento = 50000

    lucro_esperado = receita_esperada - custo_operacional
    roi_esperado = (lucro_esperado / investimento) * 100

    st.metric("ROI Esperado", f"{roi_esperado:.2f}%")

    st.markdown("""
    ### Simulação Monte Carlo para Análise de Incerteza

    Realizamos 1000 simulações variando a receita, modelada como uma distribuição normal com média R$ 80.000 e desvio padrão R$ 10.000.
    """)

    np.random.seed(42)
    sim_receita = np.random.normal(loc=80000, scale=10000, size=1000)
    sim_lucro = sim_receita - custo_operacional
    sim_roi = (sim_lucro / investimento) * 100

    df_sim = pd.DataFrame({"Receita": sim_receita, "Lucro": sim_lucro, "ROI (%)": sim_roi})

    fig_roi = px.histogram(df_sim, x="ROI (%)", nbins=40, title="Distribuição Simulada de ROI",
                           labels={"ROI (%)": "ROI Simulado (%)"})
    st.plotly_chart(fig_roi, use_container_width=True)

    prob_baixa = np.mean(sim_receita < 60000)
    st.metric("Probabilidade de Receita < R$ 60.000", f"{prob_baixa*100:.2f}%")

    st.markdown("---")
    st.subheader("Cenários de ROI")
    st.write(f"**Cenário Otimista (ROI Máximo):** {np.max(sim_roi):.2f}%")
    st.write(f"**Cenário Pessimista (ROI Mínimo):** {np.min(sim_roi):.2f}%")
    st.write(f"**Cenário Médio (ROI Médio):** {np.mean(sim_roi):.2f}%")

    if np.mean(sim_roi) > 0:
        st.success("O investimento no sistema apresenta potencial positivo.")
    else:
        st.warning("O sistema apresenta risco considerável de prejuízo. Recomenda-se análise adicional.")
