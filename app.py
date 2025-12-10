import requests
from bs4 import BeautifulSoup
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
# FUNÇÕES DE COTAÇÃO AUTOMÁTICA DO CARBONO E CÂMBIO
# =============================================================================

def obter_cotacao_carbono_investing():
    """
    Obtém a cotação em tempo real do carbono via web scraping do Investing.com
    """
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
        
        # Várias estratégias para encontrar o preço
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
                    # Remover caracteres não numéricos exceto ponto
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        # Tentativa alternativa: procurar por padrões numéricos no HTML
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
                    if 50 < preco < 200:  # Faixa razoável para carbono
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    """
    Obtém a cotação em tempo real do carbono - usa apenas Investing.com
    """
    # Tentar via Investing.com
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    # Fallback para valor padrão
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    """
    Obtém a cotação em tempo real do Euro em relação ao Real Brasileiro
    """
    try:
        # API do BCB
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        # Fallback para API alternativa
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    # Fallback para valor de referência
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    """
    Calcula o valor financeiro das emissões evitadas baseado no preço do carbono
    """
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    """
    Exibe a cotação do carbono com informações - ATUALIZADA AUTOMATICAMENTE
    """
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    # Atualização automática na primeira execução
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    # Botão para atualizar cotações
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    # Mostrar mensagem de atualização se necessário
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        # Obter cotação do carbono
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        
        # Obter cotação do Euro
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        # Atualizar session state
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        # Resetar flags
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    # Exibe cotação atual do carbono
    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    # Exibe cotação atual do Euro
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

# Inicializar todas as variáveis de session state necessárias
def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        # Buscar cotação automaticamente na inicialização
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        # Buscar cotação do Euro automaticamente
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False
    if 'executar_simulacao' not in st.session_state:
        st.session_state.executar_simulacao = False
    if 'estudo_selecionado' not in st.session_state:
        st.session_state.estudo_selecionado = 'ji_et_al'

# Chamar a inicialização
inicializar_session_state()

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO BRASILEIRA
# =============================================================================

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if pd.isna(numero):
        return "N/A"
    
    # Arredonda para 2 casas decimais
    numero = round(numero, 2)
    
    # Formata como string e substitui o ponto pela vírgula
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função de formatação para os gráficos
def br_format(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro)
    """
    if x == 0:
        return "0"
    
    # Para valores muito pequenos, usa notação científica
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    
    # Para valores grandes, formata com separador de milhar
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Para valores menores, mostra duas casas decimais
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format_5_dec(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro com 5 decimais)
    """
    return f"{x:,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# CONSTANTES E PARÂMETROS DO ARTIGO
# =============================================================================

# Dados dos artigos (Ji et al., 2013 e Shakoor et al., 2018)
DADOS_ARTIGOS = {
    'ji_et_al': {
        'nome': 'Ji et al. (2013) - Sistema Arroz',
        'emissao_convencional': 81.4,  # mg N m⁻²
        'emissao_crf': 69.6,  # mg N m⁻²
        'area': 'm²',
        'conversao_ha': 0.01,  # para converter m² para ha (fator de área)
        'reducao_percentual': 14.5,
        'reducao_rendimento': -5.0,  # % redução no rendimento
        'cultura': 'Arroz',
        'sistema': 'Monocultura',
        'rendimento_base': 7.0,  # ton/ha (valor típico para arroz)
        'preco_produto': 1500,  # R$/ton (preço médio do arroz)
        'unidade_rendimento': 'ton/ha (arroz)'
    },
    'shakoor_et_al': {
        'nome': 'Shakoor et al. (2018) - Sistema Arroz-Trigo',
        'emissao_convencional': 2.86,  # kg N ha⁻¹ (já convertido)
        'emissao_crf': 2.10,  # kg N ha⁻¹
        'area': 'ha',
        'conversao_ha': 1.0,
        'reducao_percentual': 26.5,
        'aumento_rendimento': 3.0,  # % aumento no rendimento
        'cultura': 'Arroz-Trigo',
        'sistema': 'Rotação',
        'rendimento_base': 10.0,  # ton/ha (soma arroz + trigo)
        'preco_produto': 1350,  # R$/ton (preço médio ponderado)
        'unidade_rendimento': 'ton/ha (arroz + trigo)'
    }
}

# Parâmetros econômicos (valores de mercado)
PRECO_UREIA = 1500  # R$/tonelada (preço médio)
PRECO_CRF = 2500    # R$/tonelada (preço médio, ajustado conforme dados)
DOSAGEM_N = 240     # kg N ha⁻¹ (dosagem típica)

# Fatores de conversão
FATOR_N_PARA_N2O = 44/28  # 1,571 (conversão de N para N2O)
GWP_N2O = 273  # Potencial de aquecimento global do N2O (100 anos)

# =============================================================================
# FUNÇÕES DE CÁLCULO
# =============================================================================

def converter_emissao_para_tCO2eq(emissao_kg_N_ha, area_ha):
    """
    Converte emissões de N (kg N/ha) para tCO₂eq
    
    Fórmula: kg N/ha * 1.571 (N→N₂O) / 1000 (kg→t) * 273 (GWP)
    """
    # Converter N para N₂O
    emissao_n2o_t = (emissao_kg_N_ha * FATOR_N_PARA_N2O) / 1000
    
    # Converter para CO₂eq
    tco2eq = emissao_n2o_t * GWP_N2O
    
    # Total para a área
    total_tco2eq = tco2eq * area_ha
    
    return total_tco2eq, tco2eq

def calcular_custo_fertilizante(tipo, area_ha):
    """
    Calcula custo anual dos fertilizantes
    
    ureia: 46% N
    CRF: 42% N (exemplo do artigo)
    """
    if tipo.lower() == 'convencional':
        kg_ureia = DOSAGEM_N / 0.46  # kg de ureia por ha (46% N)
        custo_ha = (kg_ureia / 1000) * PRECO_UREIA
    else:  # CRF
        kg_crf = DOSAGEM_N / 0.42  # kg de CRF per ha (42% N)
        custo_ha = (kg_crf / 1000) * PRECO_CRF
    
    custo_total = custo_ha * area_ha
    
    return custo_total, custo_ha

def calcular_rendimento(tipo, rendimento_base, area_ha, estudo):
    """
    Calcula rendimento ajustado baseado no tipo de fertilizante
    """
    dados = DADOS_ARTIGOS[estudo]
    
    if tipo.lower() == 'convencional':
        fator_ajuste = 1.0
    else:  # CRF
        if estudo == 'ji_et_al':
            fator_ajuste = 1 + (dados['reducao_rendimento'] / 100)  # -5% no Ji et al.
        else:  # shakoor_et_al
            fator_ajuste = 1 + (dados['aumento_rendimento'] / 100)  # +3% no Shakoor et al.
    
    rendimento_ajustado_ha = rendimento_base * fator_ajuste
    rendimento_total = rendimento_ajustado_ha * area_ha
    
    return rendimento_total, rendimento_ajustado_ha

def calcular_receita_carbono(reducao_tco2eq, preco_carbono, taxa_cambio=1):
    """
    Calcula receita potencial com créditos de carbono
    """
    receita_eur = reducao_tco2eq * preco_carbono
    receita_real = receita_eur * taxa_cambio
    
    return receita_real, receita_eur

def analise_viabilidade_economica(dados_simulacao):
    """
    Realiza análise de viabilidade econômica completa
    """
    resultados = {
        'fluxo_caixa': [],
        'vpl': 0,
        'tir': 0,
        'payback': 0
    }
    
    try:
        # Valores padrão para evitar KeyError
        custo_convencional_ha = dados_simulacao.get('custo_convencional_ha', 0)
        custo_crf_ha = dados_simulacao.get('custo_crf_ha', 0)
        receita_carbono_ha = dados_simulacao.get('receita_carbono_ha', 0)
        rendimento_adicional_ha = dados_simulacao.get('rendimento_adicional_ha', 0)
        preco_produto = dados_simulacao.get('preco_produto', 1000)
        
        # Calcula custo adicional do CRF
        custo_adicional_ha = custo_crf_ha - custo_convencional_ha
        
        # Calcula benefícios
        beneficio_rendimento_ha = rendimento_adicional_ha * preco_produto
        
        # Fluxo de caixa anual por hectare
        fluxo_anual_ha = receita_carbono_ha + beneficio_rendimento_ha - custo_adicional_ha
        
        # Para anos simulados
        anos = dados_simulacao.get('anos', 10)
        taxa_desconto = dados_simulacao.get('taxa_desconto', 0.06)
        
        for ano in range(1, anos + 1):
            fluxo_descontado = fluxo_anual_ha / ((1 + taxa_desconto) ** ano)
            resultados['fluxo_caixa'].append(fluxo_descontado)
        
        resultados['vpl'] = sum(resultados['fluxo_caixa'])
        
        # Payback simples
        acumulado = 0
        resultados['payback'] = anos + 1  # Valor padrão se não atingir payback
        
        for ano, fluxo in enumerate(resultados['fluxo_caixa'], 1):
            acumulado += fluxo
            if acumulado >= 0 and resultados['payback'] == anos + 1:
                resultados['payback'] = ano
                break
                
    except Exception as e:
        st.error(f"Erro na análise de viabilidade: {e}")
        resultados = {
            'fluxo_caixa': [0] * dados_simulacao.get('anos', 10),
            'vpl': 0,
            'tir': 0,
            'payback': dados_simulacao.get('anos', 10) + 1
        }
    
    return resultados

# =============================================================================
# FUNÇÕES DE SIMULAÇÃO MONTE CARLO (SEPARADAS COMO NO SCRIPTV2)
# =============================================================================

def gerar_parametros_mc(n):
    """
    Gera parâmetros para simulação Monte Carlo
    """
    np.random.seed(50)
    
    # Variações nos parâmetros principais
    preco_carbono_vals = np.random.normal(
        st.session_state.preco_carbono,
        st.session_state.preco_carbono * 0.3,  # 30% de variação
        n
    )
    
    reducao_emissao_vals = np.random.normal(
        dados_simulacao['reducao_kg_N'],
        dados_simulacao['reducao_kg_N'] * 0.2,  # 20% de variação
        n
    )
    
    if dados_simulacao['estudo'] == 'ji_et_al':
        impacto_rendimento_vals = np.random.normal(
            dados_simulacao['reducao_rendimento'],
            abs(dados_simulacao['reducao_rendimento']) * 0.1,
            n
        )
    else:
        impacto_rendimento_vals = np.random.normal(
            dados_simulacao['aumento_rendimento'],
            abs(dados_simulacao['aumento_rendimento']) * 0.1,
            n
        )
    
    return preco_carbono_vals, reducao_emissao_vals, impacto_rendimento_vals

def executar_simulacao_mc(parametros):
    """
    Executa uma simulação Monte Carlo individual
    """
    preco_carbono, reducao_emissao, impacto_rendimento = parametros
    
    # Converter redução de emissões para tCO₂eq
    reducao_tco2eq, _ = converter_emissao_para_tCO2eq(reducao_emissao, 1)
    
    # Calcular receita do carbono
    receita_carbono = calcular_receita_carbono(
        reducao_tco2eq,
        preco_carbono,
        st.session_state.taxa_cambio
    )[0]
    
    # Calcular custos
    custo_conv, _ = calcular_custo_fertilizante('convencional', 1)
    custo_crf, _ = calcular_custo_fertilizante('crf', 1)
    custo_adicional = custo_crf - custo_conv
    
    # Calcular benefício do rendimento
    beneficio_rendimento = 0
    if dados_simulacao['estudo'] == 'shakoor_et_al':
        rendimento_base = dados_simulacao.get('rendimento_base', 10)
        aumento = impacto_rendimento / 100
        beneficio_rendimento = rendimento_base * aumento * dados_simulacao.get('preco_produto', 1350)
    
    # Resultado líquido por hectare
    resultado_ha = receita_carbono + beneficio_rendimento - custo_adicional
    
    return resultado_ha

# =============================================================================
# FUNÇÕES DE ANÁLISE DE SENSIBILIDADE SOBOL (SEPARADAS COMO NO SCRIPTV2)
# =============================================================================

def definir_problema_sobol():
    """
    Define o problema para análise de sensibilidade Sobol
    """
    problema = {
        'num_vars': 3,
        'names': ['Preço Carbono (€)', 'Redução Emissões (kg N/ha)', 'Impacto Rendimento (%)'],
        'bounds': [
            [max(50, st.session_state.preco_carbono * 0.5), st.session_state.preco_carbono * 2.0],
            [0.1, 2.0],
            [-10.0, 10.0]
        ]
    }
    return problema

def executar_simulacao_sobol(parametros):
    """
    Executa uma simulação para análise Sobol
    """
    preco_carbono, reducao_emissao, impacto_rendimento = parametros
    
    # Converter redução de emissões para tCO₂eq
    reducao_tco2eq, _ = converter_emissao_para_tCO2eq(reducao_emissao, 1)
    
    # Calcular receita do carbono
    receita_carbono = calcular_receita_carbono(
        reducao_tco2eq,
        preco_carbono,
        st.session_state.taxa_cambio
    )[0]
    
    # Calcular custos
    custo_conv, _ = calcular_custo_fertilizante('convencional', 1)
    custo_crf, _ = calcular_custo_fertilizante('crf', 1)
    custo_adicional = custo_crf - custo_conv
    
    # Calcular benefício do rendimento
    beneficio_rendimento = 0
    rendimento_base = dados_simulacao.get('rendimento_base', 10)
    beneficio_rendimento = rendimento_base * (impacto_rendimento / 100) * dados_simulacao.get('preco_produto', 1350)
    
    # Resultado líquido por hectare
    resultado_ha = receita_carbono + beneficio_rendimento - custo_adicional
    
    return resultado_ha

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    st.title("🌾 Simulador de Fertilizantes Nitrogenados")
    st.markdown("""
    ### Análise de Viabilidade para Substituição de Fertilizantes Convencionais por Fertilizantes de Liberação Controlada
    
    **Baseado nos estudos:**
    - Ji et al. (2013): Sistema arroz com MSA (Mid-Season Aeration)
    - Shakoor et al. (2018): Sistema rotação arroz-trigo
    
    **Objetivo:** Analisar a viabilidade econômica e ambiental da transição
    """)
    
    # Sidebar com parâmetros
    with st.sidebar:
        # Seção de cotação do carbono - AGORA ATUALIZADA AUTOMATICAMENTE
        exibir_cotacao_carbono()
        
        st.header("⚙️ Configuração da Simulação")
        
        # Seleção do estudo base
        estudo_selecionado = st.selectbox(
            "📚 Estudo de Referência",
            options=list(DADOS_ARTIGOS.keys()),
            format_func=lambda x: DADOS_ARTIGOS[x]['nome'],
            key='estudo_selectbox'
        )
        
        # Atualizar session state quando o estudo muda
        if estudo_selecionado != st.session_state.estudo_selecionado:
            st.session_state.estudo_selecionado = estudo_selecionado
            st.rerun()
        
        # Obter dados do estudo selecionado
        dados_estudo = DADOS_ARTIGOS[estudo_selecionado]
        
        # Mostrar informações do estudo
        with st.expander(f"📖 {dados_estudo['cultura']}"):
            st.write(f"**Sistema:** {dados_estudo['sistema']}")
            st.write(f"**Redução de emissões:** {dados_estudo['reducao_percentual']}%")
            if estudo_selecionado == 'ji_et_al':
                st.write(f"**Impacto no rendimento:** {dados_estudo['reducao_rendimento']}%")
            else:
                st.write(f"**Impacto no rendimento:** +{dados_estudo['aumento_rendimento']}%")
        
        # Parâmetros gerais
        st.subheader("📍 Parâmetros da Cultura")
        
        # Rendimento base (ajustado conforme estudo)
        rendimento_base = st.slider(
            f"Rendimento Base ({dados_estudo['unidade_rendimento']})",
            min_value=float(max(1.0, dados_estudo['rendimento_base'] * 0.5)),
            max_value=float(dados_estudo['rendimento_base'] * 2.0),
            value=float(dados_estudo['rendimento_base']),
            step=0.5,
            help=f"Rendimento médio com fertilizante convencional - {dados_estudo['cultura']}"
        )
        
        # Preço do produto (ajustado conforme estudo)
        preco_produto = st.slider(
            f"Preço do {dados_estudo['cultura'].split('-')[0]} (R$/ton)",
            min_value=int(dados_estudo['preco_produto'] * 0.5),
            max_value=int(dados_estudo['preco_produto'] * 2.0),
            value=int(dados_estudo['preco_produto']),
            step=50,
            help=f"Preço de mercado do produto - {dados_estudo['cultura']}"
        )
        
        st.subheader("🏢 Parâmetros da Operação")
        
        area_total = st.slider(
            "Área Total (hectares)",
            min_value=10,
            max_value=10000,
            value=100,
            step=10,
            help="Área total cultivada"
        )
        
        anos_simulacao = st.slider(
            "Período de Simulação (anos)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="Horizonte temporal da análise"
        )
        
        taxa_desconto = st.slider(
            "Taxa de Desconto (%)",
            min_value=1.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="Taxa utilizada para descontar fluxos futuros"
        ) / 100
        
        # Configuração de simulação
        st.subheader("🎯 Configuração de Simulação")
        n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50,
                                 help="Número de iterações para análise de incerteza")
        n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16,
                             help="Número de amostras para análise de sensibilidade")
        
        # Botão de execução
        if st.button("🚀 Executar Simulação Completa", type="primary", use_container_width=True):
            st.session_state.executar_simulacao = True
    
    # Inicializar variáveis de sessão
    if 'executar_simulacao' not in st.session_state:
        st.session_state.executar_simulacao = False
    
    # Executar simulação quando solicitado
    if st.session_state.executar_simulacao:
        with st.spinner('Executando simulação...'):
            # =================================================================
            # 1. CÁLCULOS BÁSICOS
            # =================================================================
            # Obter emissões
            if dados_estudo['area'] == 'm²':
                # Converter de mg N m⁻² para kg N ha⁻¹
                emissao_conv_kg = dados_estudo['emissao_convencional'] * 0.01  # mg→kg * m²→ha
                emissao_crf_kg = dados_estudo['emissao_crf'] * 0.01
            else:
                emissao_conv_kg = dados_estudo['emissao_convencional']
                emissao_crf_kg = dados_estudo['emissao_crf']
            
            # Calcular redução de emissões
            reducao_kg_N = emissao_conv_kg - emissao_crf_kg
            reducao_tco2eq_total, reducao_tco2eq_ha = converter_emissao_para_tCO2eq(reducao_kg_N, area_total)
            
            # Calcular custos dos fertilizantes
            custo_convencional, custo_conv_ha = calcular_custo_fertilizante('convencional', area_total)
            custo_crf, custo_crf_ha = calcular_custo_fertilizante('crf', area_total)
            
            # Calcular rendimentos
            rendimento_conv, rendimento_conv_ha = calcular_rendimento(
                'convencional', rendimento_base, area_total, estudo_selecionado
            )
            rendimento_crf, rendimento_crf_ha = calcular_rendimento(
                'crf', rendimento_base, area_total, estudo_selecionado
            )
            
            # Calcular receita do carbono
            receita_carbono_real, receita_carbono_eur = calcular_receita_carbono(
                reducao_tco2eq_total,
                st.session_state.preco_carbono,
                st.session_state.taxa_cambio
            )
            
            # Calcular receita por hectare
            receita_carbono_ha = receita_carbono_real / area_total if area_total > 0 else 0
            
            # Calcular rendimento adicional por hectare
            rendimento_adicional_ha = rendimento_crf_ha - rendimento_conv_ha
            
            # =================================================================
            # 2. ANÁLISE DE VIABILIDADE
            # =================================================================
            global dados_simulacao
            dados_simulacao = {
                'anos': anos_simulacao,
                'area_ha': area_total,
                'reducao_kg_N': reducao_kg_N,
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
            
            # Adicionar dados específicos do estudo
            if estudo_selecionado == 'ji_et_al':
                dados_simulacao['reducao_rendimento'] = dados_estudo['reducao_rendimento']
            else:
                dados_simulacao['aumento_rendimento'] = dados_estudo['aumento_rendimento']
            
            # Executar análise de viabilidade
            resultados_viabilidade = analise_viabilidade_economica(dados_simulacao)
            
            # =================================================================
            # 3. APRESENTAÇÃO DOS RESULTADOS
            # =================================================================
            st.header("📈 Resultados da Simulação")
            
            # Cabeçalho com informações do estudo
            st.info(f"""
            **📋 Configuração da Simulação:**
            - **Estudo:** {dados_estudo['nome']}
            - **Cultura:** {dados_estudo['cultura']}
            - **Sistema:** {dados_estudo['sistema']}
            - **Área:** {formatar_br(area_total)} ha
            - **Período:** {anos_simulacao} anos
            - **Taxa de desconto:** {formatar_br(taxa_desconto * 100)}%
            """)
            
            # NOVA SEÇÃO: VALOR FINANCEIRO DAS EMISSÕES EVITADAS
            st.subheader("💰 Valor Financeiro das Emissões Evitadas")
            
            # Calcular valores financeiros em Euros e Reais
            valor_emissoes_eur = calcular_valor_creditos(reducao_tco2eq_total, st.session_state.preco_carbono, "€")
            valor_emissoes_brl = calcular_valor_creditos(reducao_tco2eq_total, st.session_state.preco_carbono, "R$", st.session_state.taxa_cambio)
            
            # Primeira linha: Euros
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Preço Carbono (Euro)", 
                    f"€ {formatar_br(st.session_state.preco_carbono)}/tCO₂eq",
                    help="Preço do carbono em Euros"
                )
            with col2:
                st.metric(
                    "Redução de Emissões", 
                    f"{formatar_br(reducao_tco2eq_total)} tCO₂eq",
                    help=f"Total acumulado em {anos_simulacao} anos"
                )
            with col3:
                st.metric(
                    "Valor das Reduções (Euro)", 
                    f"€ {formatar_br(valor_emissoes_eur)}",
                    help=f"Valor das emissões evitadas em Euros"
                )
            
            # Segunda linha: Reais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Preço Carbono (R$)", 
                    f"R$ {formatar_br(st.session_state.preco_carbono * st.session_state.taxa_cambio)}/tCO₂eq",
                    help="Preço do carbono convertido para Reais"
                )
            with col2:
                st.metric(
                    "Taxa de Câmbio", 
                    f"R$ {formatar_br(st.session_state.taxa_cambio)}",
                    help="1 Euro = R$ " + formatar_br(st.session_state.taxa_cambio)
                )
            with col3:
                st.metric(
                    "Valor das Reduções (R$)", 
                    f"R$ {formatar_br(valor_emissoes_brl)}",
                    help=f"Valor das emissões evitadas em Reais"
                )
            
            # =================================================================
            # 4. ANÁLISE DE SENSIBILIDADE GLOBAL (SOBOL)
            # =================================================================
            st.subheader("🎯 Análise de Sensibilidade Global (Sobol)")
            
            # Definir problema Sobol
            problema_sobol = definir_problema_sobol()
            
            # Gerar amostras Sobol
            param_values_sobol = sample(problema_sobol, n_samples)
            
            # Executar simulações em paralelo
            with st.spinner('Executando análise de sensibilidade Sobol...'):
                results_sobol = Parallel(n_jobs=-1)(
                    delayed(executar_simulacao_sobol)(params) 
                    for params in param_values_sobol
                )
            
            # Analisar resultados
            Si_sobol = analyze(problema_sobol, np.array(results_sobol), print_to_console=False)
            
            # Criar DataFrame com resultados
            sensibilidade_df = pd.DataFrame({
                'Parâmetro': problema_sobol['names'],
                'S1': Si_sobol['S1'],
                'ST': Si_sobol['ST']
            }).sort_values('ST', ascending=False)
            
            # Gráfico de barras horizontal (igual ao scriptv2)
            fig_sobol, ax_sobol = plt.subplots(figsize=(10, 6))
            sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df, palette='viridis', ax=ax_sobol)
            ax_sobol.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total)')
            ax_sobol.set_xlabel('Índice ST')
            ax_sobol.set_ylabel('')
            ax_sobol.grid(axis='x', linestyle='--', alpha=0.7)
            ax_sobol.xaxis.set_major_formatter(FuncFormatter(br_format))
            
            st.pyplot(fig_sobol)
            
            # =================================================================
            # 5. ANÁLISE DE INCERTEZA (MONTE CARLO)
            # =================================================================
            st.subheader("🎲 Análise de Incerteza (Monte Carlo)")
            
            # Gerar parâmetros para Monte Carlo
            preco_carbono_vals, reducao_emissao_vals, impacto_rendimento_vals = gerar_parametros_mc(n_simulations)
            
            # Executar simulações Monte Carlo
            with st.spinner(f'Executando {n_simulations} simulações Monte Carlo...'):
                resultados_mc = []
                for i in range(n_simulations):
                    params_mc = [
                        max(10, preco_carbono_vals[i]),  # Preço mínimo de €10
                        max(0.01, reducao_emissao_vals[i]),  # Redução mínima de 0.01 kg/ha
                        impacto_rendimento_vals[i]
                    ]
                    resultado = executar_simulacao_mc(params_mc)
                    resultados_mc.append(resultado)
            
            resultados_array_mc = np.array(resultados_mc)
            media_mc = np.mean(resultados_array_mc)
            intervalo_95_mc = np.percentile(resultados_array_mc, [2.5, 97.5])
            
            # Gráfico de histograma (igual ao scriptv2)
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            sns.histplot(resultados_array_mc, kde=True, bins=30, color='skyblue', ax=ax_mc)
            ax_mc.axvline(media_mc, color='red', linestyle='--', 
                         label=f'Média: R$ {formatar_br(media_mc)}/ha')
            ax_mc.axvline(intervalo_95_mc[0], color='green', linestyle=':', label='IC 95%')
            ax_mc.axvline(intervalo_95_mc[1], color='green', linestyle=':')
            ax_mc.set_title('Distribuição do Resultado Líquido (Simulação Monte Carlo)')
            ax_mc.set_xlabel('Resultado Líquido por Hectare (R$/ha)')
            ax_mc.set_ylabel('Frequência')
            ax_mc.legend()
            ax_mc.grid(alpha=0.3)
            ax_mc.xaxis.set_major_formatter(FuncFormatter(br_format))
            
            st.pyplot(fig_mc)
            
            # =================================================================
            # 6. ANÁLISE DE VIABILIDADE ECONÔMICA
            # =================================================================
            st.subheader("💰 Análise de Viabilidade Econômica")
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Redução de Emissões",
                    f"{formatar_br(reducao_tco2eq_total)} tCO₂eq",
                    delta=f"{dados_estudo['reducao_percentual']}%"
                )
            
            with col2:
                st.metric(
                    "Receita Carbono Potencial",
                    f"R$ {formatar_br(receita_carbono_real)}",
                    f"€ {formatar_br(receita_carbono_eur)}"
                )
            
            with col3:
                st.metric(
                    "Custo Adicional CRF",
                    f"R$ {formatar_br(custo_crf - custo_convencional)}",
                    f"{((custo_crf_ha/custo_conv_ha)-1)*100:.1f}% mais caro"
                )
            
            # Gráfico de Fluxo de Caixa
            fig_fluxo, ax_fluxo = plt.subplots(figsize=(10, 6))
            anos_array = list(range(1, anos_simulacao + 1))
            ax_fluxo.bar(anos_array, resultados_viabilidade['fluxo_caixa'])
            ax_fluxo.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax_fluxo.set_xlabel('Ano')
            ax_fluxo.set_ylabel('Fluxo de Caixa (R$/ha)')
            ax_fluxo.set_title('Fluxo de Caixa Descontado por Hectare')
            ax_fluxo.grid(True, alpha=0.3)
            ax_fluxo.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            st.pyplot(fig_fluxo)
            
            # =================================================================
            # 7. RESUMO ESTATÍSTICO
            # =================================================================
            st.subheader("📋 Resumo Estatístico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Monte Carlo")
                st.metric(
                    "Probabilidade de Viabilidade",
                    f"{(np.sum(resultados_array_mc > 0) / n_simulations) * 100:.1f}%",
                    help="Percentual de simulações onde resultado > 0"
                )
                
                st.metric(
                    "Resultado Médio",
                    f"R$ {formatar_br(media_mc)}/ha",
                    help="Resultado líquido médio por hectare"
                )
                
                st.metric(
                    "Intervalo de Confiança 95%",
                    f"[R$ {formatar_br(intervalo_95_mc[0])}, R$ {formatar_br(intervalo_95_mc[1])}]",
                    help="Intervalo de confiança do resultado"
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
                
                # Análise do preço mínimo do carbono
                if resultados_viabilidade['vpl'] < 0:
                    custo_adicional_ha = custo_crf_ha - custo_conv_ha
                    beneficio_rendimento_ha = max(0, (rendimento_crf_ha - rendimento_conv_ha) * preco_produto)
                    
                    # Calcular preço mínimo do carbono para viabilidade
                    reducao_ha = reducao_tco2eq_total / area_total
                    if reducao_ha > 0:
                        preco_minimo_ha = (custo_adicional_ha - beneficio_rendimento_ha) / reducao_ha
                        preco_minimo_eur = preco_minimo_ha / st.session_state.taxa_cambio
                        
                        st.metric(
                            "Preço Mínimo do Carbono para Viabilidade",
                            f"€ {formatar_br(preco_minimo_eur)}/tCO₂eq",
                            f"R$ {formatar_br(preco_minimo_ha)}/tCO₂eq"
                        )
            
            # =================================================================
            # 8. CONCLUSÕES E RECOMENDAÇÕES
            # =================================================================
            st.subheader("🎯 Conclusões e Recomendações")
            
            vpl_ha = resultados_viabilidade['vpl']
            probabilidade_viabilidade = (np.sum(resultados_array_mc > 0) / n_simulations) * 100
            
            if vpl_ha > 0:
                st.success(f"""
                **✅ PROJETO VIÁVEL**
                
                - **VPL positivo:** R$ {formatar_br(vpl_ha * area_total)} (R$ {formatar_br(vpl_ha)}/ha)
                - **Probabilidade de sucesso:** {probabilidade_viabilidade:.1f}%
                - **Payback:** {resultados_viabilidade['payback']} anos
                
                **Recomendações para {dados_estudo['cultura']}:**
                1. Implementar projeto piloto em área reduzida
                2. Buscar certificação VCS ou Gold Standard
                3. Negociar contratos de venda antecipada de créditos
                4. Aproveitar ganhos de produtividade (se aplicável)
                """)
            else:
                st.warning(f"""
                **⚠️ PROJETO NÃO VIÁVEL NO CENÁRIO ATUAL**
                
                - **VPL negativo:** R$ {formatar_br(vpl_ha * area_total)} (R$ {formatar_br(vpl_ha)}/ha)
                - **Probabilidade de viabilidade:** {probabilidade_viabilidade:.1f}%
                - **Fator limitante:** Custo adicional do CRF
                
                **Estratégias para viabilizar {dados_estudo['cultura']}:**
                1. Buscar subsídios governamentais para transição
                2. Negociar desconto com fornecedores de CRF
                3. Esperar aumento no preço do carbono
                4. Focar no aumento de produtividade como principal benefício
                """)
    
    else:
        # Tela inicial
        st.info("""
        ### 💡 Como usar este simulador:
        
        1. **Ajuste a cotação do carbono** na barra lateral (atualizada automaticamente)
        2. **Selecione o estudo base** na barra lateral (Ji et al. 2013 ou Shakoor et al. 2018)
        3. **Configure os parâmetros** da sua operação (área, rendimento, preços, taxa de desconto)
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
        
        # Mostrar comparação dos estudos
        st.subheader("📚 Comparação dos Estudos Base")
        
        comparacao_data = []
        for key, dados in DADOS_ARTIGOS.items():
            comparacao_data.append({
                'Estudo': dados['nome'],
                'Cultura': dados['cultura'],
                'Sistema': dados['sistema'],
                'Redução de Emissões': f"{dados['reducao_percentual']}%",
                'Impacto no Rendimento': f"{dados.get('reducao_rendimento', dados.get('aumento_rendimento', 0))}%",
                'Rendimento Base': f"{dados['rendimento_base']} {dados['unidade_rendimento']}",
                'Preço do Produto': f"R$ {formatar_br(dados['preco_produto'])}/ton"
            })
        
        df_comparacao = pd.DataFrame(comparacao_data)
        st.dataframe(df_comparacao)

if __name__ == "__main__":
    main()
