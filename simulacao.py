import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binom, poisson, norm

# Criação de abas
tab1, tab2, tab3, tab4 = st.tabs([
    "Distribuição Binomial", 
    "Distribuição Poisson", 
    "Distribuição Normal", 
    "Simulação de ROI"
])

# --- QUESTÃO 1: Distribuição Binomial - Overbooking ---
with tab1:
    st.header("Distribuição Binomial: Análise de Overbooking")

    st.markdown("""
    **Contexto:** Uma companhia aérea vendeu **130 passagens** para um voo que possui apenas **120 assentos disponíveis**, confiando que nem todos os passageiros comparecerão. A probabilidade de comparecimento é de **88%**.

    A pergunta é: **Qual o risco de overbooking?** E: **é financeiramente seguro vender 10 passagens a mais?**
    """)

    vendidos = 130
    p = 0.88
    capacidade = 120

    prob_overbooking = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("Probabilidade de Overbooking (>120 passageiros)", f"{prob_overbooking*100:.2f}%")

    st.markdown("""
    Resultado: Há uma **probabilidade de 9,24%** de que mais de 120 passageiros apareçam.
    Em termos de operação, isso representa um risco relativamente alto, podendo gerar necessidade de indenizações.
    """)

    st.markdown("---")
    st.subheader("Análise de Limite de Risco (7%)")

    limite_passagens = np.arange(120, 151)
    riscos = [1 - binom.cdf(capacidade, n, p) for n in limite_passagens]
    df_riscos = pd.DataFrame({"Passagens Vendidas": limite_passagens, "Risco de Overbooking": riscos})

    fig_risco = px.line(df_riscos, x="Passagens Vendidas", y="Risco de Overbooking",
                        title="Probabilidade de Overbooking conforme Número de Passagens Vendidas")
    fig_risco.add_hline(y=0.07, line_dash="dash", line_color="red")
    st.plotly_chart(fig_risco, use_container_width=True)

    max_seguro = df_riscos[df_riscos["Risco de Overbooking"] <= 0.07]["Passagens Vendidas"].max()
    if not np.isnan(max_seguro):
        st.success(f"Até {max_seguro} passagens podem ser vendidas sem ultrapassar o risco de 7%.")
    else:
        st.error("Nenhuma quantidade de vendas atende o limite de risco desejado.")

    st.markdown("---")
    st.subheader("Análise Financeira da Venda de 10 Passagens Extras")

    custo_indenizacao = 500
    custo_esperado = prob_overbooking * custo_indenizacao * (vendidos - capacidade)
    lucro_extra = 10 * 500

    st.metric("Lucro Bruto", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado com Overbooking", f"R$ {custo_esperado:,.2f}".replace(",", "."))

    if lucro_extra > custo_esperado:
        st.success("A venda de 10 passagens extras é financeiramente recomendada, apesar do risco moderado.")
    else:
        st.warning("O custo potencial supera o ganho. Estratégia não recomendada sem ajustes.")

# --- DISTRIBUIÇÃO POISSON ---
with tab2:
    st.header("Distribuição de Poisson - Chegada de Clientes")

    st.markdown("""
    Simulamos a chegada de clientes a uma loja utilizando a distribuição de Poisson, que modela o número de eventos por unidade de tempo.
    """)

    lambda_val = st.slider("Taxa média de chegada de clientes por hora", 1, 20, 5)
    horas = np.arange(0, 15)
    probabilidades = poisson.pmf(horas, mu=lambda_val)
    df_poisson = pd.DataFrame({"Número de Clientes": horas, "Probabilidade": probabilidades})

    fig_poisson = px.bar(df_poisson, x="Número de Clientes", y="Probabilidade",
                         title="Distribuição de Chegada de Clientes por Hora")
    st.plotly_chart(fig_poisson, use_container_width=True)

# --- DISTRIBUIÇÃO NORMAL ---
with tab3:
    st.header("Distribuição Normal - Vendas de Produtos")

    st.markdown("""
    Analisamos a variação de vendas de produtos com uma distribuição normal, ajustando média e desvio-padrão.
    Selecione um intervalo para verificar a probabilidade de vendas dentro dessa faixa.
    """)

    media = st.slider("Média de Vendas", 50, 150, 100)
    desvio = st.slider("Desvio Padrão", 5, 30, 15)
    lb = st.slider("Limite Inferior", media-40, media, media-20)
    ub = st.slider("Limite Superior", media, media+40, media+20)

    x = np.linspace(media - 4*desvio, media + 4*desvio, 400)
    y = norm.pdf(x, media, desvio)
    df_normal = pd.DataFrame({"Quantidade Vendida": x, "Densidade": y})

    fig_normal = px.area(df_normal, x="Quantidade Vendida", y="Densidade",
                         title="Distribuição Normal das Vendas")
    st.plotly_chart(fig_normal, use_container_width=True)

    prob_faixa = norm.cdf(ub, media, desvio) - norm.cdf(lb, media, desvio)
    st.success(f"Probabilidade de vendas entre {lb} e {ub} unidades: {prob_faixa*100:.2f}%")

# --- QUESTÃO 2: SIMULAÇÃO ROI ---
with tab4:
    st.header("Simulação de ROI - Novo Sistema de Informação")

    st.markdown("""
    Avaliamos o retorno do investimento (ROI) de um novo sistema, considerando incertezas nos ganhos.
    """)

    receita_esperada = 80000
    custo_operacional = 10000
    investimento = 50000

    lucro_esperado = receita_esperada - custo_operacional
    roi_esperado = (lucro_esperado / investimento) * 100

    st.metric("ROI Esperado", f"{roi_esperado:.2f}%")

    st.markdown("### Simulação Monte Carlo")

    np.random.seed(42)
    sim_receita = np.random.normal(loc=80000, scale=10000, size=1000)
    sim_lucro = sim_receita - custo_operacional
    sim_roi = (sim_lucro / investimento) * 100

    df_sim = pd.DataFrame({"Receita": sim_receita, "Lucro": sim_lucro, "ROI (%)": sim_roi})

    fig_roi = px.histogram(df_sim, x="ROI (%)", nbins=40, title="Distribuição Simulada de ROI (%)")
    st.plotly_chart(fig_roi, use_container_width=True)

    st.subheader("Análise Crítica do ROI")

    prob_baixa = np.mean(sim_receita < 60000)
    st.metric("Probabilidade da Receita ser < R$ 60.000", f"{prob_baixa*100:.2f}%")

    st.write(f"**Cenário mais otimista (ROI máximo):** {np.max(sim_roi):.2f}%")
    st.write(f"**Cenário mais pessimista (ROI mínimo):** {np.min(sim_roi):.2f}%")
    st.write(f"**ROI Médio (cenário realista):** {np.mean(sim_roi):.2f}%")

    if np.mean(sim_roi) > 0:
        st.success("Com ROI médio positivo, o investimento no sistema é aconselhável, mesmo com riscos moderados.")
    else:
        st.warning("A alta variabilidade indica risco de prejuízo. Recomenda-se cautela na implementação.")
