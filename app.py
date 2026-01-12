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
st.set_page_config(page_title="Simulador de tCO₂eq para fertilizantes nitrogenados", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO BRASILEIRA (EXATAMENTE COMO NO SCRIPTV2)
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
    
    # Informações adicionais
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
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False

# Chamar a inicialização
inicializar_session_state()

# =============================================================================
# CONSTANTES E PARÂMETROS DO ARTIGO
# =============================================================================

# Dados dos artigos (Ji et al., 2013; Shakoor et al., 2018; Zhang et al., 2025)
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
        'sistema': 'Monocultura'
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
        'sistema': 'Rotação'
    },
    'zhang_et_al_2025': {
        'nome': 'Zhang et al. (2025) - Sistema Trigo em Solos Salino-Alcalinos',
        'emissao_convencional': 0.91,  # kg N ha⁻¹ (convertido de N₂O)
        'emissao_crf': 0.37,  # kg N ha⁻¹ (convertido de N₂O)
        'area': 'ha',
        'conversao_ha': 1.0,
        'reducao_percentual': 59.4,
        'aumento_rendimento': 11.5,  # % aumento no rendimento
        'cultura': 'Trigo',
        'sistema': 'Solos Salino-Alcalinos (CRF duas aplicações)'
    }
}

# Fatores de conversão (mantidos fixos baseados nos artigos)
FATOR_N_PARA_N2O = 44/28  # 1,571 (conversão de N para N2O)
GWP_N2O = 273  # Potencial de aquecimento global do N2O (100 anos)

# Teores de nitrogênio nos fertilizantes (baseado nos artigos)
TEOR_N_UREIA = 0.46  # 46% N na ureia
TEOR_N_CRF = 0.42    # 42% N no CRF (exemplo do artigo)

# =============================================================================
# FUNÇÕES DE CÁLCULO ATUALIZADAS
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

def calcular_custo_fertilizante(tipo, area_ha, preco_ureia, preco_crf, dosagem_n):
    """
    Calcula custo anual dos fertilizantes
    
    Args:
        tipo: 'convencional' ou 'crf'
        area_ha: área em hectares
        preco_ureia: R$/ton (da sidebar)
        preco_crf: R$/ton (da sidebar)
        dosagem_n: kg N/ha (da sidebar)
    """
    if tipo.lower() == 'convencional':
        kg_ureia = dosagem_n / TEOR_N_UREIA  # kg de ureia por ha (46% N)
        custo_ha = (kg_ureia / 1000) * preco_ureia
    else:  # CRF
        kg_crf = dosagem_n / TEOR_N_CRF  # kg de CRF per ha (42% N)
        custo_ha = (kg_crf / 1000) * preco_crf
    
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
        else:  # shakoor_et_al ou zhang_et_al_2025
            fator_ajuste = 1 + (dados['aumento_rendimento'] / 100)  # +3% no Shakoor et al., +11.5% no Zhang et al.
    
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
# FUNÇÕES DE SIMULAÇÃO MONTE CARLO
# =============================================================================

def simulacao_monte_carlo(params_base, n_simulacoes=1000):
    """
    Executa simulação Monte Carlo para análise de incerteza
    """
    resultados = {
        'reducoes_tco2eq': [],
        'vpl': [],
        'viabilidade': []
    }
    
    for i in range(n_simulacoes):
        # Adicionar incerteza aos parâmetros
        params = params_base.copy()
        
        # Incerteza nas emissões (±20%)
        params['emissao_convencional'] = np.random.normal(
            params_base['emissao_convencional'],
            params_base['emissao_convencional'] * 0.2
        )
        params['emissao_crf'] = np.random.normal(
            params_base['emissao_crf'],
            params_base['emissao_crf'] * 0.2
        )
        
        # Incerteza no preço do carbono (±30%)
        params['preco_carbono'] = np.random.normal(
            params_base['preco_carbono'],
            params_base['preco_carbono'] * 0.3
        )
        
        # Incerteza no rendimento (±10%)
        if 'aumento_rendimento' in params:
            params['aumento_rendimento'] = np.random.normal(
                params_base['aumento_rendimento'],
                abs(params_base['aumento_rendimento']) * 0.1
            )
        
        # Incerteza nos preços dos fertilizantes (±15%)
        params['preco_ureia'] = np.random.normal(
            params_base['preco_ureia'],
            params_base['preco_ureia'] * 0.15
        )
        params['preco_crf'] = np.random.normal(
            params_base['preco_crf'],
            params_base['preco_crf'] * 0.15
        )
        
        # Recalcular resultados
        reducao_ha = converter_emissao_para_tCO2eq(
            params['emissao_convencional'] - params['emissao_crf'],
            1  # 1 hectare para cálculo por ha
        )[0]
        
        receita_ha = calcular_receita_carbono(
            reducao_ha,
            params['preco_carbono'],
            params.get('taxa_cambio', 5.5)
        )[0]
        
        # Usar preços com incerteza
        custo_convencional_ha = calcular_custo_fertilizante(
            'convencional', 1, 
            params['preco_ureia'], params['preco_crf'], 
            params.get('dosagem_n', 240)
        )[1]
        custo_crf_ha = calcular_custo_fertilizante(
            'crf', 1, 
            params['preco_ureia'], params['preco_crf'], 
            params.get('dosagem_n', 240)
        )[1]
        custo_adicional_ha = custo_crf_ha - custo_convencional_ha
        
        # Benefício de rendimento (se aplicável)
        beneficio_rendimento_ha = 0
        if params.get('estudo') in ['shakoor_et_al', 'zhang_et_al_2025']:
            rendimento_base = params.get('rendimento_base', 5)  # ton/ha
            aumento = params.get('aumento_rendimento', 3) / 100
            beneficio_rendimento_ha = rendimento_base * aumento * params.get('preco_produto', 1000)
        
        # Fluxo anual por hectare
        fluxo_anual_ha = receita_ha + beneficio_rendimento_ha - custo_adicional_ha
        
        # VPL simplificado (5 anos, taxa 6%)
        vpl = sum([fluxo_anual_ha / (1.06 ** (ano+1)) for ano in range(5)])
        
        resultados['reducoes_tco2eq'].append(reducao_ha)
        resultados['vpl'].append(vpl)
        resultados['viabilidade'].append(1 if vpl > 0 else 0)
    
    return resultados

def analise_sensibilidade_sobol(problema, n_amostras=100):
    """
    Análise de sensibilidade usando método Sobol
    """
    # Definir parâmetros para análise
    param_values = sample(problema, n_amostras)
    
    # Função modelo para Sobol
    def modelo_sobol(parametros):
        # Extrair parâmetros
        preco_carbono, aumento_rendimento, diferenca_emissao, custo_adicional = parametros
        
        # Calcular resultado
        reducao_tco2eq = diferenca_emissao * FATOR_N_PARA_N2O / 1000 * GWP_N2O
        receita_carbono = reducao_tco2eq * preco_carbono * 5.5  # Convertido para R$
        beneficio_rendimento = aumento_rendimento * 1000  # Valorização simplificada
        
        resultado = receita_carbono + beneficio_rendimento - custo_adicional
        
        return resultado
    
    # Executar simulações
    resultados = []
    for params in param_values:
        resultados.append(modelo_sobol(params))
    
    # Analisar resultados
    si = analyze(problema, np.array(resultados), print_to_console=False)
    
    return si, param_values, resultados

# =============================================================================
# TABELA COMPARATIVA DOS EQUIPAMENTOS E MÉTODOS DOS ARTIGOS
# =============================================================================

def criar_tabela_comparativa_artigos():
    """
    Cria tabela comparativa detalhada dos equipamentos e métodos dos três artigos
    """
    dados_comparativos = {
        'Artigo': ['Zhang et al. (2025)', 'Ji et al. (2013)', 'Shakoor et al. (2018)'],
        'Cultura': ['Trigo (solo salino-alcalino)', 'Arroz (paddy)', 'Rotações Arroz-Trigo'],
        'Duração': ['2 anos (2023-2025)', '4 anos (2008-2011)', '4 anos (2012-2015)'],
        'Método Medição N₂O': [
            'Câmara estática fechada (manual)',
            'Câmara estática (manual)',
            'Câmara estática fechada (manual)'
        ],
        'Material Câmara': [
            'Aço inoxidável (base) + Acrílico transparente (corpo)',
            'Material não especificado',
            'Poliéster (corpo) + PVC (base)'
        ],
        'Dimensões Câmara': ['50×50×150 cm', 'Não especificado', '100×50×50 cm'],
        'Amostragem Gás': [
            'Seringas gas-tight 50 mL, semanal (7 dias)',
            'Frascos de vácuo 18 mL, 2-7 dias (variável)',
            'Seringas plástico 50 mL, 3-7 dias (variável)'
        ],
        'Horário Amostragem': ['9:00-11:00 h', '8:00-12:00 h', '8:00-11:00 h'],
        'Cromatógrafo': ['Agilent 7890B', 'Shimadzu GC-14B', 'Bruker 450-GC'],
        'Detector N₂O': ['ECD (Electron Capture Detector)', 'ECD', 'Ni63ECD'],
        'Temperatura Detector': ['Não especificado', 'Não especificado', '300°C'],
        'Parâmetros Ambientais': [
            'EC solo, atividades enzimáticas (NR, NiR), fotossíntese (LI-6400)',
            'Eh (potencial redox), temperatura solo (5,10,15 cm), nível água, amostrador Rhizon',
            'Temperatura ar/solo, precipitação, WFPS, condutividade elétrica'
        ],
        'Frequência Amostragem': [
            'Semanal fixa',
            'Variável: 2-3 dias (pós-fertilização), 5 dias (outros), 7 dias (final)',
            'Variável: 3,5,7 dias conforme fase'
        ],
        'Amostras por Coleta': ['4 (0,10,20,30 min)', '4 (0,10,20,30 min)', '3 (intervalos 6 min)'],
        'Área Estudo': ['Solo salino-alcalino (EC 4.6-4.9 dS/m)', 'Arroz irrigado (MSA)', 'Rotações arroz-trigo (Chaohu)'],
        'Redução N₂O': ['59,4% (CRF duas aplicações)', '13% (média 4 anos)', '26,5% (SRF vs convencional)'],
        'Impacto Rendimento': ['+11,5%', '-5%', '+3%'],
        'Custo Amostragem': ['Alto (análises enzimáticas)', 'Moderado (amostras água)', 'Baixo-Moderado'],
        'Limitações': ['Não mencionadas', 'Amostragem manual infrequente', 'Variação sazonal significativa']
    }
    
    df_comparativo = pd.DataFrame(dados_comparativos)
    return df_comparativo

def exibir_detalhes_metodologicos():
    """
    Exibe detalhes metodológicos dos artigos em uma seção expandida
    """
    st.header("🔬 Detalhes Metodológicos dos Artigos Científicos")
    
    # Criar tabela comparativa
    df_comparativo = criar_tabela_comparativa_artigos()
    
    # Exibir tabela com formatação
    st.subheader("📋 Tabela Comparativa dos Métodos de Medição de N₂O")
    
    # Estilizar a tabela
    styled_df = df_comparativo.style.set_properties(**{
        'background-color': '#f8f9fa',
        'border': '1px solid #dee2e6',
        'font-size': '12px'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#343a40'), 
                                   ('color', 'white'),
                                   ('font-weight', 'bold'),
                                   ('text-align', 'center'),
                                   ('font-size', '13px')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#e9ecef')]}
    ])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Seção expandível com detalhes de cada artigo
    st.subheader("📚 Detalhes Específicos por Artigo")
    
    # Zhang et al. (2025)
    with st.expander("Zhang et al. (2025) - Sistema Trigo em Solos Salino-Alcalinos", expanded=True):
        st.markdown("""
        **📊 Método Principal:** Static Closed Chamber Method
        **🌱 Sistema:** Trigo em solos salino-alcalinos (EC 4.6-4.9 dS/m)
        
        **🧪 Equipamentos Específicos:**
        - **Câmara:** Base de aço inoxidável (50×50×15 cm) + corpo de acrílico (50×50×150 cm)
        - **Amostragem:** Seringas gas-tight de 50 mL, 4 amostras por coleta (0,10,20,30 min)
        - **Frequência:** Semanal durante toda a estação de crescimento
        - **Horário:** 9:00-11:00 AM
        
        **🔬 Análise Laboratorial:**
        - **Cromatógrafo:** Agilent 7890B (Agilent Technologies, USA)
        - **Detector:** Electron Capture Detector (ECD)
        - **Gás de arrasto:** N₂ (300 mL/min) para N₂O
        
        **🌡️ Parâmetros Complementares:**
        - **Solo:** Condutividade elétrica (EC meter DDS-307)
        - **Enzimas:** Atividade de nitrato redutase (NR) e nitrito redutase (NiR)
        - **Fotossíntese:** Sistema portátil LI-6400 (LICOR)
        - **Plantas:** Análise de biomassa, peso de 1000 grãos
        
        **📈 Principais Resultados:**
        - **Redução N₂O:** ~59% com CRF (duas aplicações) vs convencional
        - **Rendimento:** +11,5% com CRF vs convencional
        - **Emissões pico:** Dois picos distintos - perfilhamento/alongamento
        - **Intensidade emissão:** 0,07 kg N₂O t⁻¹ grão (CRF) vs 0,20 (convencional)
        
        **🎯 Conclusão:** CRF com duas aplicações otimiza redução de emissões e rendimento
        """)
    
    # Ji et al. (2013)
    with st.expander("Ji et al. (2013) - Sistema Arroz com MSA (Mid-Season Aeration)"):
        st.markdown("""
        **📊 Método Principal:** Static Chamber Technique (manual)
        **🌱 Sistema:** Arroz irrigado com aeração de meia estação (MSA)
        
        **🧪 Equipamentos Específicos:**
        - **Câmara:** 9 câmaras (3 tratamentos × 3 repetições), ventiladores internos
        - **Amostragem:** Frascos de vácuo de 18 mL, 4 amostras por coleta (0,10,20,30 min)
        - **Frequência:** Variável: 2-3 dias (pós-fertilização/MSA), ~5 dias (outros), 7 dias (final)
        - **Horário:** 8:00-12:00 h
        
        **🔬 Análise Laboratorial:**
        - **Cromatógrafo:** Shimadzu GC-14B (Kyoto, Japan)
        - **Detector:** Electron Capture Detector (ECD)
        - **Amostras água:** Amostrador Rhizon, armazenamento -5°C
        
        **🌡️ Parâmetros Complementares:**
        - **Solo:** Potencial redox (Eh), temperatura (5,10,15 cm), nível água
        - **Água poros:** Amostras para NH₄⁺-N e NO₃⁻-N dissolvidos
        - **Umidade solo:** Amostras 0-15 cm secas 105°C/8h
        
        **📈 Principais Resultados:**
        - **Redução N₂O:** 13% média 4 anos (CRF vs ureia)
        - **Rendimento:** -5% com CRF vs ureia
        - **Timing MSA crítico:** MSA D30 otimiza redução, MSA D40 aumenta emissões
        - **FIE (Fertilizer-induced emission):** 0,31-1,19% N aplicado
        
        **⚠️ Limitação:** Método manual infrequente pode subestimar/sobrestimar picos
        **🎯 Conclusão:** Timing da aeração (MSA) é fator crítico para otimização
        """)
    
    # Shakoor et al. (2018)
    with st.expander("Shakoor et al. (2018) - Sistema Rotação Arroz-Trigo"):
        st.markdown("""
        **📊 Método Principal:** Static Closed Chamber Method
        **🌱 Sistema:** Rotação arroz-trigo (Chaohu, China)
        
        **🧪 Equipamentos Específicos:**
        - **Câmara:** Poliéster (corpo) + PVC (base 0,5×0,5×0,15 m), 3 câmaras/parcela
        - **Amostragem:** Seringas plástico 50 mL, 3 amostras por coleta (intervalos 6 min)
        - **Frequência:** Variável: 3,5,7 dias conforme fase da cultura
        - **Horário:** 8:00-11:00 AM
        - **Recobrimento:** Folha alumínio para controle térmico
        
        **🔬 Análise Laboratorial:**
        - **Cromatógrafo:** Bruker 450-GC (USA)
        - **Detector N₂O:** Ni63ECD a 300°C
        - **Detector CH₄:** FID a 300°C
        - **Gás arrasto:** N₂ (300 mL/min) para N₂O, He para CH₄
        
        **🌡️ Parâmetros Complementares:**
        - **Clima:** Temperatura ar, precipitação (estação meteorológica)
        - **Solo:** Temperatura (0-10 cm), WFPS (water-filled pore space)
        - **Condutividade:** EC meter para solo
        - **CH₄:** Medido para cálculo GWP completo
        
        **📈 Principais Resultados:**
        - **Redução N₂O:** 26,5% com SRF (Slow-release fertilizer) vs convencional
        - **Rendimento:** +3% com SRF vs convencional
        - **Emissões variação:** 0,61 a 1707,08 µg m⁻² h⁻¹
        - **GWP reduzido:** 16,94-21,20% (SRF e OF+UI)
        
        **📊 Métricas Adicionais:**
        - **GHGI (Greenhouse Gas Intensity):** 0,16-1,20 kg CO₂-eq kg⁻¹ grão
        - **Fase principal emissão:** Crescimento vegetativo (57-81% total)
        
        **🎯 Conclusão:** SRF e OF+UI otimizam rendimento e reduzem emissões
        """)
    
    # Comparação de cromatógrafos
    st.subheader("⚖️ Comparação Técnica dos Cromatógrafos Gasosos")
    
    cromatografia_data = {
        'Modelo': ['Agilent 7890B', 'Shimadzu GC-14B', 'Bruker 450-GC'],
        'Fabricante': ['Agilent Technologies', 'Shimadzu Corporation', 'Bruker Corporation'],
        'Ano Lançamento': ['~2010', '~1990', '~2015'],
        'Precisão': ['Alta (±0,1 ppm)', 'Média (±0,5 ppm)', 'Alta (±0,2 ppm)'],
        'Detector N₂O': ['ECD moderno', 'ECD básico', 'Ni63ECD especializado'],
        'Automação': ['Alta (autosampler)', 'Baixa (manual)', 'Média'],
        'Custo Estimado': ['US$ 40-60k', 'US$ 15-25k', 'US$ 30-50k'],
        'Adequação Estudo': ['Alta resolução', 'Adequado manual', 'Balanceado']
    }
    
    df_cromatografia = pd.DataFrame(cromatografia_data)
    st.dataframe(df_cromatografia, use_container_width=True)
    
    # Recomendações metodológicas
    st.subheader("🎯 Recomendações Metodológicas para Futuros Estudos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔄 Melhorias na Amostragem:**
        - Automatização câmaras
        - Frequência diária (especialmente pós-fertilização)
        - Monitoramento contínuo 24h
        - Sensores in situ
        """)
    
    with col2:
        st.markdown("""
        **🔬 Análise Laboratorial:**
        - Cromatógrafos com autosampler
        - Padronização métodos
        - Controles de qualidade
        - Calibração frequente
        """)
    
    with col3:
        st.markdown("""
        **📊 Parâmetros Complementares:**
        - Micrometeorologia (eddy covariance)
        - Isótopos estáveis (¹⁵N)
        - Metagenômica microbiana
        - Modelagem process-based
        """)
    
    # Citação recomendada
    st.markdown("""
    ---
    **📚 Citação Recomendada para Comparação Metodológica:**
    
    *"Para estudos comparativos de métodos de medição de N₂O em sistemas agrícolas, 
    recomenda-se a consulta aos três artigos que utilizam metodologias validadas 
    de câmara estática, mas com diferentes níveis de detalhamento e frequência 
    de amostragem."*
    """)

# =============================================================================
# INTERFACE STREAMLIT - FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    st.title("🌾 Simulador de tCO₂eq para fertilizantes nitrogenados")
    st.markdown("""
    ### Análise de Viabilidade para Substituição de Fertilizantes Convencionais por Fertilizantes de Liberação Controlada
    
    **Baseado nos estudos:**
    - Ji et al. (2013): Sistema arroz com MSA (Mid-Season Aeration)
    - Shakoor et al. (2018): Sistema rotação arroz-trigo
    - Zhang et al. (2025): Sistema trigo em solos salino-alcalinos
    
    **Objetivo:** Analisar a viabilidade econômica e ambiental da transição
    """)
    
    # Sidebar com parâmetros
    with st.sidebar:
        # Seção de cotação do carbono
        exibir_cotacao_carbono()
        
        st.header("⚙️ Configuração da Simulação")
        
        # Seleção de modo
        modo_operacao = st.radio(
            "Selecione o modo:",
            ["Simulação de Viabilidade", "Detalhes Metodológicos dos Artigos"],
            index=0
        )
        
        if modo_operacao == "Simulação de Viabilidade":
            # Seleção do estudo base
            estudo_selecionado = st.selectbox(
                "📚 Estudo de Referência",
                options=list(DADOS_ARTIGOS.keys()),
                format_func=lambda x: DADOS_ARTIGOS[x]['nome']
            )
            
            # Parâmetros gerais
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
            
            # Seção de Preços dos Fertilizantes
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
            
            # Informação adicional sobre faixas de preço
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
            
            # Configurações avançadas
            with st.expander("🔧 Parâmetros Avançados"):
                taxa_desconto = st.slider(
                    "Taxa de Desconto (%)",
                    min_value=1.0,
                    max_value=15.0,
                    value=6.0,
                    step=0.5
                ) / 100
            
            # Botão de execução
            if st.button("🚀 Executar Simulação Completa", type="primary", use_container_width=True):
                st.session_state.executar_simulacao = True
                st.session_state.modo_operacao = "simulacao"
        
        else:  # Modo Detalhes Metodológicos
            if st.button("🔬 Ver Detalhes Metodológicos", type="primary", use_container_width=True):
                st.session_state.executar_simulacao = True
                st.session_state.modo_operacao = "metodologia"
    
    # Inicializar variáveis de sessão
    if 'executar_simulacao' not in st.session_state:
        st.session_state.executar_simulacao = False
    if 'modo_operacao' not in st.session_state:
        st.session_state.modo_operacao = "simulacao"
    
    # Executar conforme modo selecionado
    if st.session_state.executar_simulacao:
        if st.session_state.modo_operacao == "metodologia":
            # Exibir seção de detalhes metodológicos
            exibir_detalhes_metodologicos()
            
            # Botão para voltar
            if st.button("⬅️ Voltar para Simulação"):
                st.session_state.executar_simulacao = False
                st.rerun()
        
        else:  # Modo simulação
            with st.spinner('Executando simulação...'):
                # =================================================================
                # 1. CÁLCULOS BÁSICOS
                # =================================================================
                dados_estudo = DADOS_ARTIGOS[estudo_selecionado]
                
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
                
                # Calcular custos dos fertilizantes (usando preços da sidebar)
                custo_convencional, custo_conv_ha = calcular_custo_fertilizante(
                    'convencional', area_total, preco_ureia, preco_crf, dosagem_n
                )
                custo_crf, custo_crf_ha = calcular_custo_fertilizante(
                    'crf', area_total, preco_ureia, preco_crf, dosagem_n
                )
                
                # Calcular rendimentos
                rendimento_conv, rendimento_conv_ha = calcular_rendimento(
                    'convencional', rendimento_base, area_total, estudo_selecionado
                )
                rendimento_crf, rendimento_crf_ha = calcular_rendimento(
                    'crf', rendimento_base, area_total, estudo_selecionado
                )
                
                # Calcular receita do carbono usando as cotações automáticas
                receita_carbono_real, receita_carbono_eur = calcular_receita_carbono(
                    reducao_tco2eq_total,
                    st.session_state.preco_carbono,  # Usando a cotação automática
                    st.session_state.taxa_cambio    # Usando a taxa de câmbio automática
                )
                
                # Calcular receita por hectare
                receita_carbono_ha = receita_carbono_real / area_total if area_total > 0 else 0
                
                # Calcular rendimento adicional por hectare
                rendimento_adicional_ha = rendimento_crf_ha - rendimento_conv_ha
                
                # =================================================================
                # 2. ANÁLISE DE VIABILIDADE
                # =================================================================
                dados_viabilidade = {
                    'anos': anos_simulacao,
                    'area_ha': area_total,
                    'emissao_convencional': emissao_conv_kg,
                    'emissao_crf': emissao_crf_kg,
                    'custo_convencional_ha': custo_conv_ha,
                    'custo_crf_ha': custo_crf_ha,
                    'receita_carbono_ha': receita_carbono_ha,
                    'preco_carbono': st.session_state.preco_carbono,  # Usando a cotação automática
                    'taxa_cambio': st.session_state.taxa_cambio,      # Usando a taxa de câmbio automática
                    'taxa_desconto': taxa_desconto,
                    'rendimento_base': rendimento_base,
                    'preco_produto': preco_produto,
                    'rendimento_adicional_ha': rendimento_adicional_ha,
                    'estudo': estudo_selecionado
                }
                
                # Adicionar dados específicos do estudo
                if estudo_selecionado == 'ji_et_al':
                    dados_viabilidade['reducao_rendimento'] = dados_estudo['reducao_rendimento']
                else:
                    dados_viabilidade['aumento_rendimento'] = dados_estudo['aumento_rendimento']
                
                # Executar análise de viabilidade
                resultados_viabilidade = analise_viabilidade_economica(dados_viabilidade)
                
                # =================================================================
                # 3. MONTE CARLO
                # =================================================================
                st.subheader("🎲 Análise de Incerteza (Monte Carlo)")
                
                params_base_mc = {
                    'emissao_convencional': emissao_conv_kg,
                    'emissao_crf': emissao_crf_kg,
                    'preco_carbono': st.session_state.preco_carbono,  # Usando a cotação automática
                    'taxa_cambio': st.session_state.taxa_cambio,      # Usando a taxa de câmbio automática
                    'estudo': estudo_selecionado,
                    'rendimento_base': rendimento_base,
                    'preco_produto': preco_produto,
                    'preco_ureia': preco_ureia,      # Usando valor da sidebar
                    'preco_crf': preco_crf,          # Usando valor da sidebar
                    'dosagem_n': dosagem_n           # Usando valor da sidebar
                }
                
                if estudo_selecionado in ['shakoor_et_al', 'zhang_et_al_2025']:
                    params_base_mc['aumento_rendimento'] = dados_estudo['aumento_rendimento']
                
                resultados_mc = simulacao_monte_carlo(params_base_mc, n_simulacoes=1000)
                
                # =================================================================
                # 4. ANÁLISE DE SENSIBILIDADE (SOBOL)
                # =================================================================
                st.subheader("📊 Análise de Sensibilidade (Sobol)")
                
                problema = {
                    'num_vars': 4,
                    'names': [
                        'Preço Carbono (€)',
                        'Aumento Rendimento (%)',
                        'Diferença Emissões (kg N/ha)',
                        'Custo Adicional (R$/ha)'
                    ],
                    'bounds': [
                        [50, 150],  # Preço carbono
                        [0, 10],    # Aumento rendimento
                        [0.1, 1.5], # Diferença emissões
                        [100, 500]  # Custo adicional
                    ]
                }
                
                si, param_values, resultados_sobol = analise_sensibilidade_sobol(problema, n_amostras=100)
                
                # =================================================================
                # 5. APRESENTAÇÃO DOS RESULTADOS
                # =================================================================
                st.header("📈 Resultados da Simulação")
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Emissões Evitadas",
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
                
                # =================================================================
                # 6. ANÁLISE DE VIABILIDADE ECONÔMICA
                # =================================================================
                st.subheader("💰 Análise de Viabilidade Econômica")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Gráfico 1: Fluxo de Caixa
                anos_array = list(range(1, anos_simulacao + 1))
                axes[0].bar(anos_array, resultados_viabilidade['fluxo_caixa'])
                axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('Ano')
                axes[0].set_ylabel('Fluxo de Caixa (R$/ha)')
                axes[0].set_title('Fluxo de Caixa Descontado')
                axes[0].grid(True, alpha=0.3)
                axes[0].yaxis.set_major_formatter(FuncFormatter(br_format))
                
                # Gráfico 2: Distribuição Monte Carlo (VPL)
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
                
                # Gráfico 3: Análise de Sensibilidade
                sensibilidade_df = pd.DataFrame({
                    'Parâmetro': problema['names'],
                    'S1': si['S1'],
                    'ST': si['ST']
                }).sort_values('ST', ascending=False)
                
                axes[2].barh(sensibilidade_df['Parâmetro'], sensibilidade_df['ST'])
                axes[2].set_xlabel('Índice de Sensibilidade Total (ST)')
                axes[2].set_title('Análise de Sensibilidade (Sobol)')
                axes[2].grid(True, alpha=0.3)
                axes[2].xaxis.set_major_formatter(FuncFormatter(br_format))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # =================================================================
                # 7. RESUMO ESTATÍSTICO
                # =================================================================
                st.subheader("📋 Resumo Estatístico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Monte Carlo (1000 simulações)")
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
                                f"R$ {formatar_br(preco_minimo_ha)}/tCO₂eq",
                                help="Preço necessário para tornar o projeto viável"
                            )
                        else:
                            st.metric(
                                "Preço Mínimo do Carbono",
                                "N/A",
                                "Redução de emissões insuficiente"
                            )
                
                # =================================================================
                # 8. ANÁLISE POR CENÁRIO
                # =================================================================
                st.subheader("🌍 Análise por Cenário")
                
                # Criar cenários usando o preço atual do carbono como base
                preco_carbono_atual = st.session_state.preco_carbono
                taxa_cambio_atual = st.session_state.taxa_cambio
                
                cenarios = [
                    {'nome': 'Cenário Atual', 'preco_carbono': preco_carbono_atual, 'taxa_cambio': taxa_cambio_atual},
                    {'nome': 'Mercado em Expansão', 'preco_carbono': preco_carbono_atual * 1.4, 'taxa_cambio': taxa_cambio_atual},
                    {'nome': 'Alta do Carbono', 'preco_carbono': preco_carbono_atual * 1.75, 'taxa_cambio': taxa_cambio_atual},
                    {'nome': 'Mercado Regulado', 'preco_carbono': preco_carbono_atual * 2.3, 'taxa_cambio': taxa_cambio_atual}
                ]
                
                resultados_cenarios = []
                for cenario in cenarios:
                    receita_cenario, _ = calcular_receita_carbono(
                        reducao_tco2eq_total,
                        cenario['preco_carbono'],
                        cenario['taxa_cambio']
                    )
                    
                    vpl_cenario = sum([
                        (receita_cenario/area_total - (custo_crf_ha - custo_conv_ha) + 
                         max(0, (rendimento_crf_ha - rendimento_conv_ha) * preco_produto)) /
                        ((1 + taxa_desconto) ** ano)
                        for ano in range(1, anos_simulacao + 1)
                    ]) * area_total
                    
                    resultados_cenarios.append({
                        'Cenário': cenario['nome'],
                        'Preço Carbono (€)': formatar_br(cenario['preco_carbono']),
                        'VPL Total (R$)': formatar_br(vpl_cenario),
                        'VPL/ha (R$)': formatar_br(vpl_cenario / area_total),
                        'Viável': 'SIM' if vpl_cenario > 0 else 'NÃO'
                    })
                
                df_cenarios = pd.DataFrame(resultados_cenarios)
                
                # Aplicar formatação condicional manualmente
                def highlight_viable(val):
                    if val == 'SIM':
                        return 'background-color: lightgreen'
                    elif val == 'NÃO':
                        return 'background-color: lightcoral'
                    return ''
                
                # Aplicar estilo
                styled_df = df_cenarios.style.applymap(highlight_viable, subset=['Viável'])
                
                # Destacar máximo e mínimo manualmente
                vpl_values = [float(str(v).replace('.', '').replace(',', '.')) if isinstance(v, str) else v for v in df_cenarios['VPL Total (R$)']]
                max_idx = vpl_values.index(max(vpl_values))
                min_idx = vpl_values.index(min(vpl_values))
                
                def highlight_max_min(row):
                    styles = [''] * len(row)
                    if row.name == max_idx:
                        styles[2] = 'background-color: lightgreen'  # Coluna VPL Total
                        styles[3] = 'background-color: lightgreen'  # Coluna VPL/ha
                    elif row.name == min_idx:
                        styles[2] = 'background-color: lightcoral'  # Coluna VPL Total
                        styles[3] = 'background-color: lightcoral'  # Coluna VPL/ha
                    return styles
                
                styled_df = styled_df.apply(highlight_max_min, axis=1)
                st.dataframe(styled_df)
                
                # =================================================================
                # 9. ANÁLISE DE SENSIBILIDADE AOS PREÇOS DOS FERTILIZANTES
                # =================================================================
                st.subheader("📊 Sensibilidade aos Preços dos Insumos")
                
                # Criar cenários de variação de preço
                variacoes = [-30, -20, -10, 0, 10, 20, 30]
                resultados_sensibilidade = []
                
                for var in variacoes:
                    preco_ureia_var = preco_ureia * (1 + var/100)
                    preco_crf_var = preco_crf * (1 + var/100)
                    
                    # Recalcular custo adicional
                    custo_conv_var = calcular_custo_fertilizante(
                        'convencional', 1, preco_ureia_var, preco_crf_var, dosagem_n
                    )[1]
                    custo_crf_var = calcular_custo_fertilizante(
                        'crf', 1, preco_ureia_var, preco_crf_var, dosagem_n
                    )[1]
                    custo_adicional = custo_crf_var - custo_conv_var
                    
                    # Calcular VPL simplificado
                    beneficio_rendimento_ha = max(0, (rendimento_crf_ha - rendimento_conv_ha) * preco_produto)
                    fluxo_anual = receita_carbono_ha + beneficio_rendimento_ha - custo_adicional
                    vpl_simplificado = sum([fluxo_anual / ((1 + taxa_desconto) ** ano) for ano in range(1, 6)])
                    
                    resultados_sensibilidade.append({
                        'Variação Preços': f"{var:+}%",
                        'Custo Ureia (R$/ha)': custo_conv_var,
                        'Custo CRF (R$/ha)': custo_crf_var,
                        'Custo Adicional (R$/ha)': custo_adicional,
                        'VPL/ha (5 anos)': vpl_simplificado
                    })
                
                df_sensibilidade = pd.DataFrame(resultados_sensibilidade)
                
                # Formatar o DataFrame
                st.dataframe(df_sensibilidade.style.format({
                    'Custo Ureia (R$/ha)': lambda x: f"R$ {formatar_br(x)}",
                    'Custo CRF (R$/ha)': lambda x: f"R$ {formatar_br(x)}",
                    'Custo Adicional (R$/ha)': lambda x: f"R$ {formatar_br(x)}",
                    'VPL/ha (5 anos)': lambda x: f"R$ {formatar_br(x)}"
                }))
                
                # =================================================================
                # 10. CONCLUSÕES E RECOMENDAÇÕES
                # =================================================================
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
                    # Calcular preço mínimo se ainda não calculado
                    if resultados_viabilidade['vpl'] < 0:
                        custo_adicional_ha = custo_crf_ha - custo_conv_ha
                        beneficio_rendimento_ha = max(0, (rendimento_crf_ha - rendimento_conv_ha) * preco_produto)
                        reducao_ha = reducao_tco2eq_total / area_total
                        
                        if reducao_ha > 0:
                            preco_minimo_ha = (custo_adicional_ha - beneficio_rendimento_ha) / reducao_ha
                            preco_minimo_eur = preco_minimo_ha / st.session_state.taxa_cambio
                    
                    st.warning(f"""
                    **⚠️ PROJETO NÃO VIÁVEL NO CENÁRIO ATUAL**
                    
                    - **VPL negativo:** R$ {formatar_br(vpl_ha * area_total)} (R$ {formatar_br(vpl_ha)}/ha)
                    - **Probabilidade de viabilidade:** {formatar_br(probabilidade_viabilidade)}%
                    - **Preço atual do carbono:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                    - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                    - **Fator limitante:** Custo adicional do CRF
                    
                    **Estratégias para viabilizar:**
                    1. Buscar subsídios governamentais para transição
                    2. Negociar desconto com fornecedores de CRF (viável a partir de R$ {formatar_br(preco_crf * 0.85 if vpl_ha < 0 else preco_crf)}/ton)
                    3. Esperar aumento no preço do carbono (viável a partir de € {formatar_br(preco_minimo_eur if 'preco_minimo_eur' in locals() else 0)}/tCO₂eq)
                    4. Focar no aumento de produtividade como principal benefício
                    5. Considerar combinação CRF + ureia para reduzir custos
                    """)
                
                # Adicionar insights específicos por estudo
                with st.expander("📚 Insights Específicos por Estudo"):
                    if estudo_selecionado == 'ji_et_al':
                        st.info(f"""
                        **Ji et al. (2013) - Sistema Arroz:**
                        - CRF reduz emissões em {formatar_br(dados_estudo['reducao_percentual'])}%, mas reduz rendimento em {formatar_br(abs(dados_estudo['reducao_rendimento']))}%
                        - Timing da aeração (MSA) é crítico: MSA em D30 otimiza redução
                        - Necessário compensar perda de rendimento com valor agregado ou carbono
                        - **Preço do carbono atual:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                        - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                        """)
                    elif estudo_selecionado == 'shakoor_et_al':
                        st.info(f"""
                        **Shakoor et al. (2018) - Sistema Arroz-Trigo:**
                        - CRF reduz emissões em {formatar_br(dados_estudo['reducao_percentual'])}% e aumenta rendimento em {formatar_br(dados_estudo['aumento_rendimento'])}%
                        - Sistema de rotação otimiza benefícios
                        - Viabilidade mais provável devido ao duplo benefício
                        - **Preço do carbono atual:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                        - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                        """)
                    else:  # zhang_et_al_2025
                        st.info(f"""
                        **Zhang et al. (2025) - Sistema Trigo em Solos Salino-Alcalinos:**
                        - CRF com duas aplicações reduz emissões em {formatar_br(dados_estudo['reducao_percentual'])}% e aumenta rendimento em {formatar_br(dados_estudo['aumento_rendimento'])}%
                        - Sistema otimizado para solos salino-alcalinos (EC 4.6-4.9 dS/m)
                        - Maior redução de emissões entre todos os estudos (59,4%)
                        - **Preço do carbono atual:** €{formatar_br(st.session_state.preco_carbono)}/tCO₂eq
                        - **Custo adicional do CRF:** R$ {formatar_br(custo_crf - custo_convencional)} ({formatar_br(((custo_crf_ha/custo_conv_ha)-1)*100)}% mais caro)
                        - **Recomendação:** Duas aplicações de CRF (50% basal + 50% na fase de perfilhamento)
                        """)
    
    else:
        # Tela inicial
        if modo_operacao == "Simulação de Viabilidade":
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
            
            # Mostrar comparação dos estudos
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
        
        else:  # Modo Detalhes Metodológicos
            st.info("""
            ### 🔬 Detalhes Metodológicos dos Artigos
            
            Esta seção apresenta uma análise comparativa detalhada dos métodos e equipamentos
            utilizados nos três artigos científicos que fundamentam este simulador.
            
            **O que você encontrará:**
            1. **Tabela comparativa completa** dos métodos de medição de N₂O
            2. **Detalhes específicos** de cada artigo
            3. **Comparação técnica** dos cromatógrafos gasosos utilizados
            4. **Recomendações metodológicas** para futuros estudos
            
            Clique no botão **"Ver Detalhes Metodológicos"** na barra lateral para acessar
            a análise completa.
            """)
            
            # Mostrar prévia da tabela comparativa
            st.subheader("📋 Prévia da Tabela Comparativa")
            df_previa = criar_tabela_comparativa_artigos()
            st.dataframe(df_previa.head(3), use_container_width=True)
            
            st.markdown("""
            **📊 Colunas da tabela completa:**
            - Artigo, Cultura, Duração do estudo
            - Método de medição de N₂O
            - Equipamentos utilizados (câmaras, amostradores)
            - Frequência e horário de amostragem
            - Equipamentos de análise laboratorial
            - Parâmetros ambientais medidos
            - Principais resultados e limitações
            """)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    main()
