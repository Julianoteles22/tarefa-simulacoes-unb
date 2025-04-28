import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binom, poisson, norm
from PIL import Image
import os

# Logo da UnB (arquivo "unb-logo.webp" na mesma pasta)
script_dir = os.path.dirname(__file__)
logo_path = os.path.join(script_dir, "trabalhar-na-UnB-1200x1200.png")
if os.path.exists(logo_path):
    img = Image.open(logo_path)
    st.image(img, width=200)
else:
    st.warning("Logo da UnB não encontrado em 'unb-logo.webp'. Verifique o nome e localização.")

# Título e alunos
st.title("Overbooking e ROI: Explorando Decisões de Dados e Lucros")
st.markdown(
    """
**Alunos:**  
- Juliano Teles Abrahao - 231013411  
- João Pedro Carvalho - 231013402
"""
)

# Criação de abas
tab1, tab2, tab3, tab4 = st.tabs([
    "Distribuição Binomial", 
    "Distribuição Poisson", 
    "Distribuição Normal", 
    "Simulação Interativa de ROI"
])

# --- QUESTÃO 1 ---
with tab1:
    st.header("Distribuição Binomial: Overbooking Aéreo")
    st.markdown("""
**Contexto Detalhado:**
A Aérea Confiável vendeu um número de passagens acima da capacidade do voo para maximizar a ocupação, apostando que nem todos os passageiros comparecerão. Este módulo interativo permite ajustar parâmetros-chave e visualizar, de forma dinâmica, o comportamento da probabilidade de overbooking e suas implicações financeiras.
""")

    vendidos = st.slider("Passagens Vendidas", 120, 200, 130)
    p = st.slider("Taxa de Comparecimento (%)", 0.0, 1.0, 0.88, step=0.01)
    capacidade = st.number_input("Capacidade do Voo", 1, 200, 120)

    xs = np.arange(vendidos+1)
    pmf = binom.pmf(xs, vendidos, p)
    df_pmf = pd.DataFrame({"Comparecimentos": xs, "Probabilidade": pmf})
    fig_pmf = px.bar(df_pmf, x="Comparecimentos", y="Probabilidade",
                     title="Distribuição Binomial de Comparecimento",
                     labels={"Probabilidade":"P(X=k)"})
    fig_pmf.add_vline(x=capacidade, line_dash="dash", line_color="red",
                      annotation_text="Capacidade")
    st.plotly_chart(fig_pmf, use_container_width=True)

    prob_overbooking = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("Probabilidade de Overbooking (> capacidade)", f"{prob_overbooking:.2%}")

    st.subheader("Limite de Risco de Overbooking (≤ 7%)")
    venda_range = np.arange(capacidade, vendidos*2+1)
    riscos = [1 - binom.cdf(capacidade, n, p) for n in venda_range]
    df_risco = pd.DataFrame({"Passagens": venda_range, "Risco": riscos})
    fig_risco = px.line(df_risco, x="Passagens", y="Risco",
                        title="Risco vs. Passagens Vendidas")
    fig_risco.add_hline(y=0.07, line_dash="dash", line_color="red",
                        annotation_text="Limite 7%")
    st.plotly_chart(fig_risco, use_container_width=True)
    max_safe = df_risco[df_risco.Risco <= 0.07].Passagens.max()
    if not np.isnan(max_safe): st.success(f"Máximo seguro de passagens: {int(max_safe)}")

    st.subheader("Viabilidade Financeira: +10 Assentos")
    custo_ind = st.number_input("Custo de Indenização (R$)", 0, 10000, 500)
    preco_med = st.number_input("Preço Médio por Passagem (R$)", 0, 10000, 500)
    lucro_ext = 10 * preco_med
    custo_esp = prob_overbooking * custo_ind * (vendidos - capacidade)
    st.metric("Lucro Bruto com 10 Extras", f"R$ {lucro_ext:,.2f}")
    st.metric("Custo Esperado Indenizações", f"R$ {custo_esp:,.2f}")
    st.markdown(
        """
**Análise Crítica:**  
- O custo esperado reduz a margem de lucro.  
- Avalie políticas de seguro e flexibilização para mitigar riscos.
"""
    )

# --- DISTRIBUIÇÃO POISSON ---
with tab2:
    st.header("Distribuição de Poisson - Chegada de Clientes")
    lam = st.slider("λ (clientes/hora)", 1, 20, 5)
    x = np.arange(0, 16)
    df_poi = pd.DataFrame({"Clientes": x, "Prob": poisson.pmf(x, lam)})
    fig_poi = px.bar(df_poi, x="Clientes", y="Prob",
                     title="Poisson PMF")
    st.plotly_chart(fig_poi, use_container_width=True)

# --- DISTRIBUIÇÃO NORMAL ---
with tab3:
    st.header("Distribuição Normal - Vendas")
    mu = st.slider("Média", 50, 150, 100)
    sd = st.slider("Desvio Padrão", 5, 30, 15)
    xx = np.linspace(mu-4*sd, mu+4*sd, 200)
    df_norm = pd.DataFrame({"Vendas": xx, "Densidade": norm.pdf(xx, mu, sd)})
    fig_norm = px.area(df_norm, x="Vendas", y="Densidade", title="Normal PDF")
    st.plotly_chart(fig_norm, use_container_width=True)

# --- SIMULAÇÃO ROI ---
with tab4:
    st.header("Simulação de ROI")
    receita = st.number_input("Receita (R$)", 0, 200000, 80000)
    custo = st.number_input("Custo Op. (R$)", 0, 50000, 10000)
    inv = st.number_input("Investimento (R$)", 0, 100000, 50000)
    sims = st.slider("Simulações", 100, 5000, 1000)
    roi = (receita - custo) / inv * 100
    st.metric("ROI Esperado", f"{roi:.2f}%")
    sim_rev = np.random.normal(receita, 0.2*receita, sims)
    sim_prof = sim_rev - custo
    sim_roi = sim_prof / inv * 100
    df_sim = pd.DataFrame({"ROI": sim_roi})
    st.plotly_chart(px.histogram(df_sim, x="ROI", nbins=40, title="Hist ROI"), use_container_width=True)
    st.plotly_chart(px.ecdf(df_sim, x="ROI", title="CDF ROI"), use_container_width=True)
    prob_neg = np.mean(sim_roi<0)
    st.metric("Prob ROI Negativo", f"{prob_neg:.2%}")
    texto = (
        f"**Cenário Otimista:** {np.max(sim_roi):.2f}%  \n"
        f"**Cenário Pessimista:** {np.min(sim_roi):.2f}%  \n"
        f"**Prob. Negativa:** {prob_neg:.2%}"
    )
    st.markdown(texto)
    if np.mean(sim_roi)>0: st.success("ROI médio positivo")
    else: st.warning("ROI médio negativo")
