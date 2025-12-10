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

np.random.seed(50)  # Garante reprodutibilidade

# Configurações iniciais
st.set_page_config(page_title="Simulador de Fertilizantes Nitrogenados", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO BRASILEIRA
# =============================================================================

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format_5_dec(x, pos):
    return f"{x:,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO
# =============================================================================

def obter_cotacao_carbono_investing():
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
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
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        import re
        padroes_preco = [
            r'"last":"([\d,]+)"',
            r'data-last="([\d,]+)"',
            r'last_price["\']?:\s*["\']?([\d,]+)',
            r'value["\']?:\s*["\']?([\d,]+)'
        ]
        
        html_texto = str(soup)
        for padrao in padroes_preco:
            matches = re.findall(padrao, html_texto)
            for match in matches:
                try:
                    preco_texto = match.replace(',', '')
                    preco = float(preco_texto)
                    if 50 < preco < 200:
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    # Inicializar variáveis se não existirem
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'preco_carbono' not in st.session_state:
        st.session_state.preco_carbono = 85.50
        st.session_state.moeda_carbono = "€"
        st.session_state.fonte_cotacao = "Referência"
    if 'taxa_cambio' not in st.session_state:
        st.session_state.taxa_cambio = 5.50
        st.session_state.moeda_real = "R$"
    
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
        
        # Garantir que temos valores válidos
        if preco_carbono is None:
            preco_carbono = 85.50
            moeda = "€"
            fonte_carbono = "Referência"
        if preco_euro is None:
            preco_euro = 5.50
            moeda_real = "R$"
        
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

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
    
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}
        - **Carbono em Reais:** R$ {formatar_br(preco_carbono_reais)}/tCO₂eq
        
        **🌍 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono
        - Preços em tempo real
        
        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        """)

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        if preco_carbono is None:
            preco_carbono = 85.50
            moeda = "€"
            fonte = "Referência"
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        if preco_euro is None:
            preco_euro = 5.50
            moeda_real = "R$"
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False

inicializar_session_state()

# =============================================================================
# CONSTANTES E PARÂMETROS DO ARTIGO
# =============================================================================

DADOS_ARTIGOS = {
    'ji_et_al': {
        'nome': 'Ji et al. (2013) - Sistema Arroz',
        'emissao_convencional': 81.4,
        'emissao_crf': 69.6,
        'area': 'm²',
        'conversao_ha': 0.01,
        'reducao_percentual': 14.5,
        'reducao_rendimento': -5.0,
        'cultura': 'Arroz',
        'sistema': 'Monocultura'
    },
    'shakoor_et_al': {
        'nome': 'Shakoor et al. (2018) - Sistema Arroz-Trigo',
        'emissao_convencional': 2.86,
        'emissao_crf': 2.10,
        'area': 'ha',
        'conversao_ha': 1.0,
        'reducao_percentual': 26.5,
        'aumento_rendimento': 3.0,
        'cultura': 'Arroz-Trigo',
        'sistema': 'Rotação'
    },
    'zhang_et_al_2025': {
        'nome': 'Zhang et al. (2025) - Sistema Trigo em Solos Salino-Alcalinos',
        'emissao_convencional': 0.91,
        'emissao_crf': 0.37,
        'area': 'ha',
        'conversao_ha': 1.0,
        'reducao_percentual': 59.4,
        'aumento_rendimento': 11.5,
        'cultura': 'Trigo',
        'sistema': 'Solos Salino-Alcalinos (CRF duas aplicações)'
    }
}

FATOR_N_PARA_N2O = 44/28
GWP_N2O = 273
TEOR_N_UREIA = 0.46
TEOR_N_CRF = 0.42

# =============================================================================
# FUNÇÕES DE CÁLCULO ATUALIZADAS
# =============================================================================

def converter_emissao_para_tCO2eq(emissao_kg_N_ha, area_ha):
    emissao_n2o_t = (emissao_kg_N_ha * FATOR_N_PARA_N2O) / 1000
    tco2eq = emissao_n2o_t * GWP_N2O
    total_tco2eq = tco2eq * area_ha
    return total_tco2eq, tco2eq

def calcular_custo_fertilizante(tipo, area_ha, preco_ureia, preco_crf, dosagem_n):
    if tipo.lower() == 'convencional':
        kg_ureia = dosagem_n / TEOR_N_UREIA
        custo_ha = (kg_ureia / 1000) * preco_ureia
    else:
        kg_crf = dosagem_n / TEOR_N_CRF
        custo_ha = (kg_crf / 1000) * preco_crf
    
    custo_total = custo_ha * area_ha
    return custo_total, custo_ha

def calcular_rendimento(tipo, rendimento_base, area_ha, estudo):
    dados = DADOS_ARTIGOS[estudo]
    
    if tipo.lower() == 'convencional':
        fator_ajuste = 1.0
    else:
        if estudo == 'ji_et_al':
            fator_ajuste = 1 + (dados['reducao_rendimento'] / 100)
        else:
            fator_ajuste = 1 + (dados['aumento_rendimento'] / 100)
    
    rendimento_ajustado_ha = rendimento_base * fator_ajuste
    rendimento_total = rendimento_ajustado_ha * area_ha
    return rendimento_total, rendimento_ajustado_ha

def calcular_receita_carbono(reducao_tco2eq, preco_carbono, taxa_cambio=1):
    receita_eur = reducao_tco2eq * preco_carbono
    receita_real = receita_eur * taxa_cambio
    return receita_real, receita_eur

def analise_viabilidade_economica(dados_simulacao):
    resultados = {
        'fluxo_caixa': [],
        'vpl': 0,
        'tir': 0,
        'payback': 0
    }
    
    try:
        custo_convencional_ha = dados_simulacao.get('custo_convencional_ha', 0)
        custo_crf_ha = dados_simulacao.get('custo_crf_ha', 0)
        receita_carbono_ha = dados_simulacao.get('receita_carbono_ha', 0)
        rendimento_adicional_ha = dados_simulacao.get('rendimento_adicional_ha', 0)
        preco_produto = dados_simulacao.get('preco_produto', 1000)
        
        custo_adicional_ha = custo_crf_ha - custo_convencional_ha
        beneficio_rendimento_ha = rendimento_adicional_ha * preco_produto
        fluxo_anual_ha = receita_carbono_ha + beneficio_rendimento_ha - custo_adicional_ha
        
        anos = dados_simulacao.get('anos', 10)
        taxa_desconto = dados_simulacao.get('taxa_desconto', 0.06)
        
        for ano in range(1, anos + 1):
            fluxo_descontado = fluxo_anual_ha / ((1 + taxa_desconto) ** ano)
            resultados['fluxo_caixa'].append(fluxo_descontado)
        
        resultados['vpl'] = sum(resultados['fluxo_caixa'])
        
        acumulado = 0
        resultados['payback'] = anos + 1
        
        for ano, fluxo in enumerate(resultados['fluxo_caixa'], 1):
            acumulado += fluxo
            if acumulado >= 0 and resultados['payback'] == anos + 1:
                resultados['payback'] = ano
                break
                
    except Exception as e:
        resultados = {
            'fluxo_caixa': [0] * dados_simulacao.get('anos', 10),
            'vpl': 0,
            'tir': 0,
            'payback': dados_simulacao.get('anos', 10) + 1
        }
    
    return resultados

# =============================================================================
# FUNÇÕES DE SIMULAÇÃO MONTE CARLO
# =============================================================================

def simulacao_monte_carlo(params_base, n_simulacoes=1000):
    resultados = {
        'reducoes_tco2eq': [],
        'vpl': [],
        'viabilidade': []
    }
    
    for i in range(n_simulacoes):
        params = params_base.copy()
        
        params['emissao_convencional'] = np.random.normal(
            params_base['emissao_convencional'],
            max(params_base['emissao_convencional'] * 0.2, 0.1)
        )
        params['emissao_crf'] = np.random.normal(
            params_base['emissao_crf'],
            max(params_base['emissao_crf'] * 0.2, 0.1)
        )
        
        params['preco_carbono'] = np.random.normal(
            params_base['preco_carbono'],
            max(params_base['preco_carbono'] * 0.3, 0.1)
        )
        
        if 'aumento_rendimento' in params:
            params['aumento_rendimento'] = np.random.normal(
                params_base['aumento_rendimento'],
                max(abs(params_base['aumento_rendimento']) * 0.1, 0.1)
            )
        
        params['preco_ureia'] = np.random.normal(
            params_base['preco_ureia'],
            max(params_base['preco_ureia'] * 0.15, 0.1)
        )
        params['preco_crf'] = np.random.normal(
            params_base['preco_crf'],
            max(params_base['preco_crf'] * 0.15, 0.1)
        )
        
        reducao_ha = converter_emissao_para_tCO2eq(
            max(params['emissao_convencional'] - params['emissao_crf'], 0),
            1
        )[0]
        
        receita_ha = calcular_receita_carbono(
            reducao_ha,
            max(params['preco_carbono'], 0),
            params.get('taxa_cambio', 5.5)
        )[0]
        
        custo_convencional_ha = calcular_custo_fertilizante(
            'convencional', 1, 
            max(params['preco_ureia'], 0), max(params['preco_crf'], 0), 
            params.get('dosagem_n', 240)
        )[1]
        custo_crf_ha = calcular_custo_fertilizante(
            'crf', 1, 
            max(params['preco_ureia'], 0), max(params['preco_crf'], 0), 
            params.get('dosagem_n', 240)
        )[1]
        custo_adicional_ha = custo_crf_ha - custo_convencional_ha
        
        beneficio_rendimento_ha = 0
        if params.get('estudo') in ['shakoor_et_al', 'zhang_et_al_2025']:
            rendimento_base = params.get('rendimento_base', 5)
            aumento = params.get('aumento_rendimento', 3) / 100
            beneficio_rendimento_ha = rendimento_base * aumento * params.get('preco_produto', 1000)
        
        fluxo_anual_ha = receita_ha + beneficio_rendimento_ha - custo_adicional_ha
        vpl = sum([fluxo_anual_ha / (1.06 ** (ano+1)) for ano in range(5)])
        
        resultados['reducoes_tco2eq'].append(reducao_ha)
        resultados['vpl'].append(vpl)
        resultados['viabilidade'].append(1 if vpl > 0 else 0)
    
    return resultados

# =============================================================================
# ANÁLISE DE SENSIBILIDADE SOBOL ATUALIZADA
# =============================================================================

def analise_sensibilidade_sobol_completa(params_base, n_amostras=100):
    problema = {
        'num_vars': 6,
        'names': [
            'Preço Carbono (€/tCO₂eq)',
            'Preço CRF (R$/ton)',
            'Dosagem N (kg/ha)',
            'Rendimento Base (ton/ha)',
            'Preço Produto (R$/ton)',
            'Área (ha)'
        ],
        'bounds': [
            [max(40, params_base.get('preco_carbono', 85) * 0.5), 
             params_base.get('preco_carbono', 85) * 1.5],
            [params_base.get('preco_crf', 2500) * 0.7, 
             params_base.get('preco_crf', 2500) * 1.3],
            [params_base.get('dosagem_n', 240) * 0.7, 
             params_base.get('dosagem_n', 240) * 1.3],
            [params_base.get('rendimento_base', 5) * 0.7, 
             params_base.get('rendimento_base', 5) * 1.3],
            [params_base.get('preco_produto', 1000) * 0.7, 
             params_base.get('preco_produto', 1000) * 1.3],
            [params_base.get('area_ha', 100) * 0.5, 
             params_base.get('area_ha', 100) * 1.5]
        ]
    }
    
    param_values = sample(problema, n_amostras)
    
    def modelo_real(parametros):
        try:
            preco_carbono, preco_crf, dosagem_n, rendimento_base, preco_produto, area_ha = parametros
            
            estudo = params_base.get('estudo', 'shakoor_et_al')
            dados_estudo = DADOS_ARTIGOS.get(estudo, DADOS_ARTIGOS['shakoor_et_al'])
            
            if dados_estudo['area'] == 'm²':
                emissao_conv_kg = dados_estudo['emissao_convencional'] * 0.01
                emissao_crf_kg = dados_estudo['emissao_crf'] * 0.01
            else:
                emissao_conv_kg = dados_estudo['emissao_convencional']
                emissao_crf_kg = dados_estudo['emissao_crf']
            
            reducao_kg_N = max(emissao_conv_kg - emissao_crf_kg, 0)
            reducao_tco2eq_total, _ = converter_emissao_para_tCO2eq(reducao_kg_N, area_ha)
            
            preco_ureia = params_base.get('preco_ureia', 1500)
            _, custo_conv_ha = calcular_custo_fertilizante(
                'convencional', area_ha, preco_ureia, preco_crf, dosagem_n
            )
            _, custo_crf_ha = calcular_custo_fertilizante(
                'crf', area_ha, preco_ureia, preco_crf, dosagem_n
            )
            
            _, rendimento_conv_ha = calcular_rendimento(
                'convencional', rendimento_base, area_ha, estudo
            )
            _, rendimento_crf_ha = calcular_rendimento(
                'crf', rendimento_base, area_ha, estudo
            )
            
            taxa_cambio = params_base.get('taxa_cambio', 5.5)
            receita_carbono_real, _ = calcular_receita_carbono(
                reducao_tco2eq_total, preco_carbono, taxa_cambio
            )
            receita_carbono_ha = receita_carbono_real / area_ha if area_ha > 0 else 0
            
            rendimento_adicional_ha = rendimento_crf_ha - rendimento_conv_ha
            beneficio_rendimento_ha = rendimento_adicional_ha * preco_produto
            
            custo_adicional_ha = custo_crf_ha - custo_conv_ha
            
            fluxo_anual_ha = receita_carbono_ha + beneficio_rendimento_ha - custo_adicional_ha
            
            taxa_desconto = 0.06
            vpl_ha = sum([fluxo_anual_ha / ((1 + taxa_desconto) ** ano) for ano in range(1, 6)])
            
            return vpl_ha
            
        except Exception as e:
            return 0
    
    resultados = []
    for params in param_values:
        resultados.append(modelo_real(params))
    
    si = analyze(problema, np.array(resultados), print_to_console=False)
    
    return si, param_values, np.array(resultados)

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    try:
        st.title("🌾 Simulador de Fertilizantes Nitrogenados")
        st.markdown("""
        ### Análise de Viabilidade para Substituição de Fertilizantes Convencionais por Fertilizantes de Liberação Controlada
        
        **Baseado nos estudos:**
        - Ji et al. (2013): Sistema arroz com MSA (Mid-Season Aeration)
        - Shakoor et al. (2018): Sistema rotação arroz-trigo
        - Zhang et al. (2025): Sistema trigo em solos salino-alcalinos
        
        **Objetivo:** Analisar a viabilidade econômica e ambiental da transição
        """)
        
        with st.sidebar:
            exibir_cotacao_carbono()
            
            st.header("⚙️ Configuração da Simulação")
            
            estudo_selecionado = st.selectbox(
                "📚 Estudo de Referência",
                options=list(DADOS_ARTIGOS.keys()),
                format_func=lambda x: DADOS_ARTIGOS[x]['nome'],
                index=0
            )
            
            area_total = st.slider(
                "Área Total (hectares)",
                min_value=10,
                max_value=10000,
                value=100,
                step=10
            )
            
            anos_simulacao = st.slider(
                "Período de Simulação (anos)",
                min_value=5,
                max_value=30,
                value=10,
                step=5
            )
            
            rendimento_base = st.slider(
                "Rendimento Base (ton/ha/ano)",
                min_value=2.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Rendimento médio com fertilizante convencional"
            )
            
            preco_produto = st.slider(
                "Preço do Produto (R$/ton)",
                min_value=500,
                max_value=2000,
                value=1000,
                step=50
            )
            
            st.subheader("💰 Preços dos Fertilizantes (R$/tonelada)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                preco_ureia = st.number_input(
                    "Ureia Convencional",
                    min_value=1000,
                    max_value=3000,
                    value=1500,
                    step=50,
                    help="Preço atual da ureia (46% N)"
                )
                
            with col2:
                preco_crf = st.number_input(
                    "Fertilizante CRF",
                    min_value=1500,
                    max_value=5000,
                    value=2500,
                    step=50,
                    help="Preço do fertilizante de liberação controlada (42% N)"
                )
            
            dosagem_n = st.slider(
                "Dosagem de Nitrogênio (kg N/ha)",
                min_value=100,
                max_value=400,
                value=240,
                step=10,
                help="Quantidade de nitrogênio aplicada por hectare"
            )
            
            with st.expander("💡 Informações sobre preços médios"):
                st.markdown("""
                **Faixas de Preço de Referência (2024):**
                
                | Fertilizante | Faixa Típica (R$/ton) | Observação |
                |--------------|----------------------|------------|
                | **Ureia** | 1.400 - 2.400 | Varia com região e época |
                | **CRF** | 2.500 - 4.500 | Depende da tecnologia/marca |
                
                **Fontes:**
                - CONAB (Companhia Nacional de Abastecimento)
                - CEPEA/ESALQ (Centro de Estudos Avançados)
                - Mercado local
                """)
            
            with st.expander("🔧 Parâmetros Avançados"):
                taxa_desconto = st.slider(
                    "Taxa de Desconto (%)",
                    min_value=1.0,
                    max_value=15.0,
                    value=6.0,
                    step=0.5
                ) / 100
            
            if st.button("🚀 Executar Simulação Completa", type="primary", use_container_width=True):
                st.session_state.executar_simulacao = True
        
        # Inicializar variáveis para evitar UnboundLocalError
        executar_simulacao = st.session_state.get('executar_simulacao', False)
        sensibilidade_significativa = pd.DataFrame()  # Inicializar variável
        
        if executar_simulacao:
            with st.spinner('Executando simulação...'):
                # Inicializar variáveis com valores padrão
                emissao_conv_kg = 0
                emissao_crf_kg = 0
                custo_convencional = 0
                custo_conv_ha = 0
                custo_crf = 0
                custo_crf_ha = 0
                rendimento_conv = 0
                rendimento_conv_ha = 0
                rendimento_crf = 0
                rendimento_crf_ha = 0
                receita_carbono_real = 0
                receita_carbono_ha = 0
                reducao_tco2eq_total = 0
                reducao_tco2eq_ha = 0
                
                dados_estudo = DADOS_ARTIGOS.get(estudo_selecionado, DADOS_ARTIGOS['shakoor_et_al'])
                
                if dados_estudo['area'] == 'm²':
                    emissao_conv_kg = dados_estudo['emissao_convencional'] * 0.01
                    emissao_crf_kg = dados_estudo['emissao_crf'] * 0.01
                else:
                    emissao_conv_kg = dados_estudo['emissao_convencional']
                    emissao_crf_kg = dados_estudo['emissao_crf']
                
                reducao_kg_N = emissao_conv_kg - emissao_crf_kg
                reducao_tco2eq_total, reducao_tco2eq_ha = converter_emissao_para_tCO2eq(reducao_kg_N, area_total)
                
                custo_convencional, custo_conv_ha = calcular_custo_fertilizante(
                    'convencional', area_total, preco_ureia, preco_crf, dosagem_n
                )
                custo_crf, custo_crf_ha = calcular_custo_fertilizante(
                    'crf', area_total, preco_ureia, preco_crf, dosagem_n
                )
                
                rendimento_conv, rendimento_conv_ha = calcular_rendimento(
                    'convencional', rendimento_base, area_total, estudo_selecionado
                )
                rendimento_crf, rendimento_crf_ha = calcular_rendimento(
                    'crf', rendimento_base, area_total, estudo_selecionado
                )
                
                receita_carbono_real, receita_carbono_eur = calcular_receita_carbono(
                    reducao_tco2eq_total,
                    st.session_state.preco_carbono,
                    st.session_state.taxa_cambio
                )
                
                receita_carbono_ha = receita_carbono_real / area_total if area_total > 0 else 0
                rendimento_adicional_ha = rendimento_crf_ha - rendimento_conv_ha
                
                dados_viabilidade = {
                    'anos': anos_simulacao,
                    'area_ha': area_total,
                    'emissao_convencional': emissao_conv_kg,
                    'emissao_crf': emissao_crf_kg,
                    'custo_convencional_ha': custo_conv_ha,
                    'custo_crf_ha': custo_crf_ha,
                    'receita_carbono_ha': receita_carbono_ha,
                    'preco_carbono': st.session_state.preco_carbono,
                    'taxa_cambio': st.session_state.taxa_cambio,
                    'taxa_desconto': taxa_desconto,
                    'rendimento_base': rendimento_base,
                    'preco_produto': preco_produto,
                    'rendimento_adicional_ha': rendimento_adicional_ha,
                    'estudo': estudo_selecionado
                }
                
                if estudo_selecionado == 'ji_et_al':
                    dados_viabilidade['reducao_rendimento'] = dados_estudo['reducao_rendimento']
                else:
                    dados_viabilidade['aumento_rendimento'] = dados_estudo['aumento_rendimento']
                
                resultados_viabilidade = analise_viabilidade_economica(dados_viabilidade)
                
                # Continuação da simulação...
                st.subheader("🎲 Análise de Incerteza (Monte Carlo)")
                
                params_base_mc = {
                    'emissao_convencional': emissao_conv_kg,
                    'emissao_crf': emissao_crf_kg,
                    'preco_carbono': st.session_state.preco_carbono,
                    'taxa_cambio': st.session_state.taxa_cambio,
                    'estudo': estudo_selecionado,
                    'rendimento_base': rendimento_base,
                    'preco_produto': preco_produto,
                    'preco_ureia': preco_ureia,
                    'preco_crf': preco_crf,
                    'dosagem_n': dosagem_n
                }
                
                if estudo_selecionado in ['shakoor_et_al', 'zhang_et_al_2025']:
                    params_base_mc['aumento_rendimento'] = dados_estudo['aumento_rendimento']
                
                resultados_mc = simulacao_monte_carlo(params_base_mc, n_simulacoes=500)
                
                st.subheader("📊 Análise de Sensibilidade (Sobol)")
                
                params_base_sobol = {
                    'preco_carbono': st.session_state.preco_carbono,
                    'preco_crf': preco_crf,
                    'preco_ureia': preco_ureia,
                    'dosagem_n': dosagem_n,
                    'rendimento_base': rendimento_base,
                    'preco_produto': preco_produto,
                    'taxa_cambio': st.session_state.taxa_cambio,
                    'area_ha': area_total,
                    'estudo': estudo_selecionado
                }
                
                try:
                    si, param_values, resultados_sobol = analise_sensibilidade_sobol_completa(
                        params_base_sobol, n_amostras=100
                    )
                    
                    sensibilidade_df = pd.DataFrame({
                        'Parâmetro': ['Preço Carbono (€/tCO₂eq)', 'Preço CRF (R$/ton)', 
                                      'Dosagem N (kg/ha)', 'Rendimento Base (ton/ha)',
                                      'Preço Produto (R$/ton)', 'Área (ha)'],
                        'S1 (Efeito Principal)': si['S1'],
                        'ST (Efeito Total)': si['ST']
                    }).sort_values('ST', ascending=False)
                    
                    sensibilidade_significativa = sensibilidade_df[sensibilidade_df['ST'] > 0.01].copy()
                    
                    fig_sensibilidade, ax = plt.subplots(figsize=(10, 6))
                    
                    if len(sensibilidade_significativa) > 0:
                        sensibilidade_significativa = sensibilidade_significativa.sort_values('ST', ascending=True)
                        y_pos = np.arange(len(sensibilidade_significativa))
                        
                        ax.barh(y_pos - 0.2, sensibilidade_significativa['S1'], height=0.4, 
                                label='Efeito Principal (S1)', alpha=0.7, color='skyblue')
                        ax.barh(y_pos + 0.2, sensibilidade_significativa['ST'], height=0.4, 
                                label='Efeito Total (ST)', alpha=0.7, color='coral')
                        
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sensibilidade_significativa['Parâmetro'])
                        ax.set_xlabel('Índice de Sensibilidade')
                        ax.set_title('Análise de Sensibilidade (Sobol) - Parâmetros Mais Influentes')
                        ax.legend()
                        
                        for i, (s1, st) in enumerate(zip(sensibilidade_significativa['S1'], 
                                                        sensibilidade_significativa['ST'])):
                            ax.text(s1 + 0.01, i - 0.2, f'{s1:.3f}', va='center', fontsize=9)
                            ax.text(st + 0.01, i + 0.2, f'{st:.3f}', va='center', fontsize=9)
                        
                        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        
                    else:
                        ax.text(0.5, 0.5, 'Nenhum parâmetro com sensibilidade significativa (> 0.01)',
                                ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        ax.set_title('Análise de Sensibilidade (Sobol)')
                    
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_sensibilidade)
                    
                except Exception as e:
                    st.warning(f"Não foi possível realizar a análise de sensibilidade: {str(e)}")
                    sensibilidade_significativa = pd.DataFrame()  # Garantir que está definida
                
                # Continuação com os resultados...
                st.header("📈 Resultados da Simulação")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Redução de Emissões",
                        f"{formatar_br(reducao_tco2eq_total)} tCO₂eq",
                        delta=f"{formatar_br(dados_estudo['reducao_percentual'])}%"
                    )
                
                with col2:
                    st.metric(
                        "Receita Carbono Potencial",
                        f"R$ {formatar_br(receita_carbono_real)}",
                        f"€ {formatar_br(receita_carbono_eur)}",
                        help=f"Preço do carbono: €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq"
                    )
                
                with col3:
                    st.metric(
                        "Custo Adicional CRF",
                        f"R$ {formatar_br(custo_crf - custo_convencional)}",
                        f"{formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro"
                    )
                
                with col4:
                    if estudo_selecionado == 'ji_et_al':
                        delta_rend = f"{formatar_br(dados_estudo['reducao_rendimento'])}%"
                    else:
                        delta_rend = f"+{formatar_br(dados_estudo['aumento_rendimento'])}%"
                    
                    st.metric(
                        "Impacto no Rendimento",
                        f"{formatar_br(rendimento_crf)} ton",
                        delta_rend
                    )
                
                # O restante do código de visualização...
                st.subheader("💰 Análise de Viabilidade Econômica")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                anos_array = list(range(1, anos_simulacao + 1))
                axes[0].bar(anos_array, resultados_viabilidade['fluxo_caixa'])
                axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('Ano')
                axes[0].set_ylabel('Fluxo de Caixa (R$/ha)')
                axes[0].set_title('Fluxo de Caixa Descontado')
                axes[0].grid(True, alpha=0.3)
                axes[0].yaxis.set_major_formatter(FuncFormatter(br_format))
                
                axes[1].hist(resultados_mc['vpl'], bins=30, edgecolor='black', alpha=0.7)
                axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Ponto de Equilíbrio')
                
                media_vpl = np.mean(resultados_mc['vpl'])
                axes[1].axvline(x=media_vpl, color='g', linestyle='-', 
                               linewidth=2, label=f'Média: R$ {formatar_br(media_vpl)}')
                
                axes[1].set_xlabel('VPL (R$/ha)')
                axes[1].set_ylabel('Frequência')
                axes[1].set_title('Distribuição do VPL (Monte Carlo)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].xaxis.set_major_formatter(FuncFormatter(br_format))
                
                if not sensibilidade_significativa.empty and len(sensibilidade_significativa) > 0:
                    top_params = sensibilidade_significativa.sort_values('ST', ascending=False).head(4)
                    axes[2].barh(top_params['Parâmetro'], top_params['ST'])
                    axes[2].set_xlabel('Índice de Sensibilidade Total (ST)')
                    axes[2].set_title('Parâmetros Mais Influentes (Sobol)')
                    
                    for i, (param, st) in enumerate(zip(top_params['Parâmetro'], top_params['ST'])):
                        axes[2].text(st + 0.01, i, f'{st:.3f}', va='center', fontsize=9)
                        
                    axes[2].axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Limite moderado (0.1)')
                    axes[2].axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Limite alto (0.3)')
                    axes[2].legend(fontsize=8)
                else:
                    axes[2].text(0.5, 0.5, 'Sensibilidade não significativa', 
                                ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('Análise de Sensibilidade (Sobol)')
                
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("📋 Resumo Estatístico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Monte Carlo (500 simulações)")
                    probabilidade = np.mean(resultados_mc['viabilidade']) * 100
                    st.metric(
                        "Probabilidade de Viabilidade",
                        f"{formatar_br(probabilidade)}%",
                        help="Percentual de simulações onde VPL > 0"
                    )
                    
                    st.metric(
                        "VPL Médio",
                        f"R$ {formatar_br(np.mean(resultados_mc['vpl']))}/ha",
                        help="Valor Presente Líquido médio por hectare"
                    )
                    
                    perc_2_5 = np.percentile(resultados_mc['vpl'], 2.5)
                    perc_97_5 = np.percentile(resultados_mc['vpl'], 97.5)
                    intervalo_texto = f"[R$ {formatar_br(perc_2_5)}, R$ {formatar_br(perc_97_5)}]"
                    
                    st.metric(
                        "Intervalo de Confiança 95%",
                        intervalo_texto,
                        help="Intervalo de confiança do VPL"
                    )
                
                with col2:
                    st.write("#### Viabilidade Base")
                    st.metric(
                        "VPL do Projeto",
                        f"R$ {formatar_br(resultados_viabilidade['vpl'] * area_total)}",
                        f"R$ {formatar_br(resultados_viabilidade['vpl'])}/ha"
                    )
                    
                    st.metric(
                        "Payback Simples",
                        f"{resultados_viabilidade['payback']} anos",
                        help="Tempo para recuperar o investimento"
                    )
                    
                    if resultados_viabilidade['vpl'] < 0:
                        custo_adicional_ha = custo_crf_ha - custo_conv_ha
                        beneficio_rendimento_ha = max(0, (rendimento_crf_ha - rendimento_conv_ha) * preco_produto)
                        reducao_ha = reducao_tco2eq_total / area_total
                        
                        if reducao_ha > 0:
                            preco_minimo_ha = (custo_adicional_ha - beneficio_rendimento_ha) / reducao_ha
                            preco_minimo_eur = preco_minimo_ha / st.session_state.taxa_cambio
                            
                            st.metric(
                                "Preço Mínimo do Carbono para Viabilidade",
                                f"€ {formatar_br(preco_minimo_eur)}/tCO₂eq",
                                f"R$ {formatar_br(preco_minimo_ha)}/tCO₂eq",
                                help="Preço necessário para tornar o projeto viável"
                            )
                        else:
                            st.metric(
                                "Preço Mínimo do Carbono",
                                "N/A",
                                "Redução de emissões insuficiente"
                            )
                
                # Resumo final
                st.subheader("🎯 Conclusões e Recomendações")
                
                vpl_ha = resultados_viabilidade['vpl']
                probabilidade_viabilidade = np.mean(resultados_mc['viabilidade']) * 100
                
                if vpl_ha > 0:
                    st.success(f"""
                    **✅ PROJETO VIÁVEL**
                    
                    - **VPL positivo:** R$ {formatar_br(vpl_ha * area_total)} (R$ {formatar_br(vpl_ha)}/ha)
                    - **Probabilidade de sucesso:** {formatar_br(probabilidade_viabilidade)}%
                    - **Payback:** {resultados_viabilidade['payback']} anos
                    - **Preço atual do carbono:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                    - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                    
                    **Recomendações:**
                    1. Implementar projeto piloto em área reduzida
                    2. Buscar certificação VCS ou Gold Standard
                    3. Negociar contratos de venda antecipada de créditos
                    4. Aproveitar ganhos de produtividade (se aplicável)
                    """)
                else:
                    st.warning(f"""
                    **⚠️ PROJETO NÃO VIÁVEL NO CENÁRIO ATUAL**
                    
                    - **VPL negativo:** R$ {formatar_br(vpl_ha * area_total)} (R$ {formatar_br(vpl_ha)}/ha)
                    - **Probabilidade de viabilidade:** {formatar_br(probabilidade_viabilidade)}%
                    - **Preço atual do carbono:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                    - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                    - **Fator limitante:** Custo adicional do CRF
                    
                    **Estratégias para viabilizar:**
                    1. Buscar subsídios governamentais para transição
                    2. Negociar desconto com fornecedores de CRF
                    3. Esperar aumento no preço do carbono
                    4. Focar no aumento de produtividade como principal benefício
                    5. Considerar combinação CRF + ureia para reduzir custos
                    """)
                
        else:
            st.info("""
            ### 💡 Como usar este simulador:
            
            1. **Acompanhe as cotações do carbono e câmbio** na seção superior da barra lateral
            2. **Selecione o estudo base** na seção de configuração (Ji et al. 2013, Shakoor et al. 2018 ou Zhang et al. 2025)
            3. **Configure os parâmetros** da sua operação (área, rendimento, preços)
            4. **Clique em "Executar Simulação Completa"**
            5. **Analise os resultados** de viabilidade econômica e ambiental
            
            ### 📊 O que será analisado:
            - Redução de emissões de N₂O
            - Custo-benefício da substituição
            - Impacto no rendimento das culturas
            - Análise de sensibilidade e incerteza
            - Cenários de preço do carbono
            - Recomendações específicas
            """)
            
            st.subheader("📚 Comparação dos Estudos Base")
            
            comparacao_data = []
            for key, dados in DADOS_ARTIGOS.items():
                comparacao_data.append({
                    'Estudo': dados['nome'],
                    'Cultura': dados['cultura'],
                    'Sistema': dados['sistema'],
                    'Emissão Convencional': f"{formatar_br(dados['emissao_convencional'])} {dados['area']}",
                    'Emissão CRF': f"{formatar_br(dados['emissao_crf'])} {dados['area']}",
                    'Redução': f"{formatar_br(dados['reducao_percentual'])}%",
                    'Impacto Rendimento': f"{formatar_br(dados.get('reducao_rendimento', dados.get('aumento_rendimento', 0)))}%"
                })
            
            df_comparacao = pd.DataFrame(comparacao_data)
            st.dataframe(df_comparacao)
    
    except Exception as error:
        st.error(f"Ocorreu um erro no aplicativo: {str(error)}")
        st.info("""
        **Solução de problemas:**
        1. Tente recarregar a página
        2. Verifique sua conexão com a internet
        3. Contate o suporte técnico se o problema persistir
        """)

if __name__ == "__main__":
    main()
