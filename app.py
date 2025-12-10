import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import requests
from bs4 import BeautifulSoup
import traceback

# ---------------------------
# Configurações iniciais
# ---------------------------
np.random.seed(50)  # Garante reprodutibilidade
st.set_page_config(page_title="Simulador de Fertilizantes Nitrogenados", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ---------------------------
# Formatação brasileira
# ---------------------------

def formatar_br(numero):
    if numero is None or (isinstance(numero, float) and np.isnan(numero)):
        return "N/A"
    try:
        numero = float(numero)
    except Exception:
        return str(numero)
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace('.', ',').replace('X', '.')


def br_format(x, pos=None):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace('.', ',')
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')


def br_format_5_dec(x, pos=None):
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{x:,.5f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# ---------------------------
# Cotação do Carbono e Câmbio
# ---------------------------

@st.cache_data(ttl=60*30)
def obter_cotacao_carbono_investing():
    """Tenta puxar preço de carbono do investing.com.
    Retorna: (preco, moeda, descricao, sucesso, fonte)
    """
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last',
            '.pid-1062510-last',
            '.float_lang_base_1',
            '.top.bold.inlineblock',
            '#last_last'
        ]

        preco = None
        fonte = "Investing.com"

        for seletor in selectores:
            elemento = soup.select_one(seletor)
            if elemento and elemento.text:
                texto_preco = elemento.text.strip().replace(',', '')
                texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                if texto_preco:
                    try:
                        preco = float(texto_preco)
                        break
                    except Exception:
                        continue

        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte

        # tentativa por regex (fallback)
        import re
        padroes_preco = [
            r'"last":"?([\d\.,]+)"?',
            r'data-last="?([\d\.,]+)"?',
            r'last_price["\']?:\s*["\']?([\d\.,]+)',
            r'value["\']?:\s*["\']?([\d\.,]+)'
        ]
        html_texto = str(soup)
        for padrao in padroes_preco:
            matches = re.findall(padrao, html_texto)
            for match in matches:
                preco_texto = match.replace(',', '')
                try:
                    preco = float(preco_texto)
                    if 10 < preco < 1000:  # sanity check
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except Exception:
                    continue

        return None, None, None, False, fonte

    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"


@st.cache_data(ttl=60*30)
def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except Exception:
        pass

    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates'].get('BRL')
            if cotacao:
                return float(cotacao), "R$", True, "ExchangeRate-API"
    except Exception:
        pass

    return 5.50, "R$", False, "Referência"


def obter_cotacao_carbono():
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    if sucesso:
        return preco, moeda, contrato_info, True, fonte
    # fallback (valor referência)
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"


def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    try:
        valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
        return valor_total
    except Exception:
        return 0


def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")

    # inicializações seguras
    st.session_state.setdefault('cotacao_carregada', False)
    st.session_state.setdefault('mostrar_atualizacao', False)
    st.session_state.setdefault('cotacao_atualizada', False)
    st.session_state.setdefault('preco_carbono', 85.50)
    st.session_state.setdefault('moeda_carbono', '€')
    st.session_state.setdefault('fonte_cotacao', 'Referência')
    st.session_state.setdefault('taxa_cambio', 5.50)
    st.session_state.setdefault('moeda_real', 'R$')

    if not st.session_state.cotacao_carregada:
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True

    if st.session_state.mostrar_atualizacao:
        st.sidebar.info("🔄 Atualizando cotações...")
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()

        if preco_carbono is None:
            preco_carbono = 85.50
            moeda = '€'
            fonte_carbono = 'Referência'
        if preco_euro is None:
            preco_euro = 5.50
            moeda_real = 'R$'

        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono

        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False

        # rerun to refresh UI with new values
        st.experimental_rerun()

    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio

    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )

    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )

    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )

    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbo
