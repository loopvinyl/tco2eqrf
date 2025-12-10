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
import locale

# Tentar configurar locale para Português Brasil
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Portuguese_Brazil.1252')
    except:
        pass

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
# FUNÇÕES AUXILIARES DE FORMATAÇÃO BRASILEIRA
# =============================================================================

def formatar_numero_brasileiro(valor, casas_decimais=2):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if valor is None or pd.isna(valor):
        return "N/A"
    
    try:
        # Converter para float se for string
        if isinstance(valor, str):
            valor = float(valor.replace(',', '.'))
        
        # Arredondar para o número de casas decimais
        valor_arredondado = round(float(valor), casas_decimais)
        
        # Formatar com separador de milhar e decimal
        if casas_decimais == 0:
            formato = "{:,.0f}"
        else:
            formato = "{:,.%df}" % casas_decimais
        
        # Aplicar formatação e substituir vírgula por ponto e ponto por vírgula
        valor_formatado = formato.format(valor_arredondado)
        
        # Substituir separadores
        if ',' in valor_formatado and '.' in valor_formatado:
            # Tem ambos separadores (milhar e decimal)
            valor_formatado = valor_formatado.replace(',', 'X').replace('.', ',').replace('X', '.')
        elif ',' in valor_formatado:
            # Tem apenas vírgula (separador decimal em inglês)
            valor_formatado = valor_formatado.replace(',', ',')
        
        return valor_formatado
    except (ValueError, TypeError):
        return str(valor)

def formatar_percentual(valor):
    """
    Formata percentuais: 14,5%
    """
    if valor is None or pd.isna(valor):
        return "N/A"
    
    try:
        if isinstance(valor, str):
            valor = float(valor.replace(',', '.'))
        
        valor_arredondado = round(float(valor), 1)
        return f"{formatar_numero_brasileiro(valor_arredondado, 1)}%"
    except (ValueError, TypeError):
        return str(valor)

# Funções de formatação para matplotlib
def formatador_br_milhares(x, pos):
    """
    Formata números para eixos de gráficos (padrão brasileiro para milhares)
    """
    if x == 0:
        return "0"
    elif abs(x) < 0.01:
        return f"{x:.1e}".replace('.', ',')
    elif abs(x) >= 1000:
        return f"{x:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    else:
        return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def formatador_br_decimal(x, pos):
    """
    Formata números para eixos de gráficos (padrão brasileiro para decimais)
    """
    return f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

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
        value=f"{st.session_state.moeda_carbono} {formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    # Exibe cotação atual do Euro
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_numero_brasileiro(st.session_state.taxa_cambio, 2)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_numero_brasileiro(preco_carbono_reais, 2)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    # Informações adicionais
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {formatar_numero_brasileiro(st.session_state.taxa_cambio, 2)}
        - **Carbono em Reais:** R$ {formatar_numero_brasileiro(preco_carbono_reais, 2)}/tCO₂eq
        
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
        
        custo_convencional_ha = calcular_custo_fertilizante('convencional', 1)[1]
        custo_crf_ha = calcular_custo_fertilizante('crf', 1)[1]
        custo_adicional_ha = custo_crf_ha - custo_convencional_ha
        
        # Benefício de rendimento (se aplicável)
        beneficio_rendimento_ha = 0
        if params.get('estudo') == 'shakoor_et_al':
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
        # Seção de cotação do carbono
        exibir_cotacao_carbono()
        
        st.header("⚙️ Configuração da Simulação")
        
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
        
        # Configurações avançadas (agora sem os campos duplicados de carbono e câmbio)
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
    
    # Inicializar variáveis de sessão
    if 'executar_simulacao' not in st.session_state:
        st.session_state.executar_simulacao = False
    
    # Executar simulação quando solicitado
    if st.session_state.executar_simulacao:
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
                'preco_produto': preco_produto
            }
            
            if estudo_selecionado == 'shakoor_et_al':
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
                    f"{formatar_numero_brasileiro(reducao_tco2eq_total, 0)} tCO₂eq",
                    delta=f"{formatar_numero_brasileiro(dados_estudo['reducao_percentual'], 1)}%"
                )
            
            with col2:
                st.metric(
                    "Receita Carbono Potencial",
                    f"R$ {formatar_numero_brasileiro(receita_carbono_real, 0)}",
                    f"€ {formatar_numero_brasileiro(receita_carbono_eur, 0)}",
                    help=f"Preço do carbono: €{formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq"
                )
            
            with col3:
                st.metric(
                    "Custo Adicional CRF",
                    f"R$ {formatar_numero_brasileiro(custo_crf - custo_convencional, 0)}",
                    f"{formatar_numero_brasileiro(((custo_crf_ha/custo_conv_ha)-1)*100, 1)}% mais caro"
                )
            
            with col4:
                if estudo_selecionado == 'ji_et_al':
                    delta_rend = f"{formatar_numero_brasileiro(dados_estudo['reducao_rendimento'], 1)}%"
                else:
                    delta_rend = f"+{formatar_numero_brasileiro(dados_estudo['aumento_rendimento'], 1)}%"
                
                st.metric(
                    "Impacto no Rendimento",
                    f"{formatar_numero_brasileiro(rendimento_crf, 0)} ton",
                    delta_rend
                )
            
            # =================================================================
            # 6. ANÁLISE DE VIABILIDADE ECONÔMICA
            # =================================================================
            st.subheader("💰 Análise de Viabilidade Econômica")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Gráfico 1: Fluxo de Caixa
            anos_array = list(range(1, anos_simulacao + 1))
            fluxo_formatado = [formatar_numero_brasileiro(f, 0) for f in resultados_viabilidade['fluxo_caixa']]
            
            # Criar barras
            bars = axes[0].bar(anos_array, resultados_viabilidade['fluxo_caixa'])
            axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0].set_xlabel('Ano')
            axes[0].set_ylabel('Fluxo de Caixa (R$/ha)')
            axes[0].set_title('Fluxo de Caixa Descontado')
            axes[0].grid(True, alpha=0.3)
            
            # Formatar eixo Y com padrão brasileiro
            axes[0].yaxis.set_major_formatter(FuncFormatter(formatador_br_milhares))
            
            # Gráfico 2: Distribuição Monte Carlo (VPL)
            axes[1].hist(resultados_mc['vpl'], bins=30, edgecolor='black', alpha=0.7)
            axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Ponto de Equilíbrio')
            
            # Calcular média formatada
            media_vpl = np.mean(resultados_mc['vpl'])
            media_vpl_formatada = formatar_numero_brasileiro(media_vpl, 0)
            axes[1].axvline(x=media_vpl, color='g', linestyle='-', 
                           linewidth=2, label=f'Média: R$ {media_vpl_formatada}')
            
            axes[1].set_xlabel('VPL (R$/ha)')
            axes[1].set_ylabel('Frequência')
            axes[1].set_title('Distribuição do VPL (Monte Carlo)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].xaxis.set_major_formatter(FuncFormatter(formatador_br_milhares))
            
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
            axes[2].xaxis.set_major_formatter(FuncFormatter(formatador_br_decimal))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # =================================================================
            # 7. RESUMO ESTATÍSTICO
            # =================================================================
            st.subheader("📋 Resumo Estatístico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Monte Carlo (1000 simulações)")
                st.metric(
                    "Probabilidade de Viabilidade",
                    f"{formatar_numero_brasileiro(np.mean(resultados_mc['viabilidade']) * 100, 1)}%",
                    help="Percentual de simulações onde VPL > 0"
                )
                
                st.metric(
                    "VPL Médio",
                    f"R$ {formatar_numero_brasileiro(np.mean(resultados_mc['vpl']), 0)}/ha",
                    help="Valor Presente Líquido médio por hectare"
                )
                
                # Calcular intervalo de confiança formatado
                perc_2_5 = np.percentile(resultados_mc['vpl'], 2.5)
                perc_97_5 = np.percentile(resultados_mc['vpl'], 97.5)
                intervalo_texto = f"[R$ {formatar_numero_brasileiro(perc_2_5, 0)}, R$ {formatar_numero_brasileiro(perc_97_5, 0)}]"
                
                st.metric(
                    "Intervalo de Confiança 95%",
                    intervalo_texto,
                    help="Intervalo de confiança do VPL"
                )
            
            with col2:
                st.write("#### Viabilidade Base")
                st.metric(
                    "VPL do Projeto",
                    f"R$ {formatar_numero_brasileiro(resultados_viabilidade['vpl'] * area_total, 0)}",
                    f"R$ {formatar_numero_brasileiro(resultados_viabilidade['vpl'], 0)}/ha"
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
                            f"€ {formatar_numero_brasileiro(preco_minimo_eur, 0)}/tCO₂eq",
                            f"R$ {formatar_numero_brasileiro(preco_minimo_ha, 0)}/tCO₂eq",
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
                    'Preço Carbono (€)': formatar_numero_brasileiro(cenario['preco_carbono'], 2),
                    'VPL Total (R$)': formatar_numero_brasileiro(vpl_cenario, 0),
                    'VPL/ha (R$)': formatar_numero_brasileiro(vpl_cenario / area_total, 0),
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
            # 9. CONCLUSÕES E RECOMENDAÇÕES
            # =================================================================
            st.subheader("🎯 Conclusões e Recomendações")
            
            vpl_ha = resultados_viabilidade['vpl']
            probabilidade_viabilidade = np.mean(resultados_mc['viabilidade']) * 100
            
            if vpl_ha > 0:
                st.success(f"""
                **✅ PROJETO VIÁVEL**
                
                - **VPL positivo:** R$ {formatar_numero_brasileiro(vpl_ha * area_total, 0)} (R$ {formatar_numero_brasileiro(vpl_ha, 0)}/ha)
                - **Probabilidade de sucesso:** {formatar_numero_brasileiro(probabilidade_viabilidade, 1)}%
                - **Payback:** {resultados_viabilidade['payback']} anos
                - **Preço atual do carbono:** €{formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq
                
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
                
                - **VPL negativo:** R$ {formatar_numero_brasileiro(vpl_ha * area_total, 0)} (R$ {formatar_numero_brasileiro(vpl_ha, 0)}/ha)
                - **Probabilidade de viabilidade:** {formatar_numero_brasileiro(probabilidade_viabilidade, 1)}%
                - **Preço atual do carbono:** €{formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq
                - **Fator limitante:** Custo adicional do CRF
                
                **Estratégias para viabilizar:**
                1. Buscar subsídios governamentais para transição
                2. Negociar desconto com fornecedores de CRF
                3. Esperar aumento no preço do carbono (viável a partir de € {formatar_numero_brasileiro(preco_minimo_eur if 'preco_minimo_eur' in locals() else 0, 0)}/tCO₂eq)
                4. Focar no aumento de produtividade como principal benefício
                5. Considerar combinação CRF + ureia para reduzir custos
                """)
            
            # Adicionar insights específicos por estudo
            with st.expander("📚 Insights Específicos por Estudo"):
                if estudo_selecionado == 'ji_et_al':
                    st.info(f"""
                    **Ji et al. (2013) - Sistema Arroz:**
                    - CRF reduz emissões em {formatar_numero_brasileiro(dados_estudo['reducao_percentual'], 1)}%, mas reduz rendimento em {formatar_numero_brasileiro(abs(dados_estudo['reducao_rendimento']), 1)}%
                    - Timing da aeração (MSA) é crítico: MSA em D30 otimiza redução
                    - Necessário compensar perda de rendimento com valor agregado ou carbono
                    - **Preço do carbono atual:** €{formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq
                    """)
                else:
                    st.info(f"""
                    **Shakoor et al. (2018) - Sistema Arroz-Trigo:**
                    - CRF reduz emissões em {formatar_numero_brasileiro(dados_estudo['reducao_percentual'], 1)}% e aumenta rendimento em {formatar_numero_brasileiro(dados_estudo['aumento_rendimento'], 1)}%
                    - Sistema de rotação otimiza benefícios
                    - Viabilidade mais provável devido ao duplo benefício
                    - **Preço do carbono atual:** €{formatar_numero_brasileiro(st.session_state.preco_carbono, 2)}/tCO₂eq
                    """)
    
    else:
        # Tela inicial
        st.info("""
        ### 💡 Como usar este simulador:
        
        1. **Acompanhe as cotações do carbono e câmbio** na seção superior da barra lateral
        2. **Selecione o estudo base** na seção de configuração (Ji et al. 2013 ou Shakoor et al. 2018)
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
                'Emissão Convencional': f"{formatar_numero_brasileiro(dados['emissao_convencional'], 2)} {dados['area']}",
                'Emissão CRF': f"{formatar_numero_brasileiro(dados['emissao_crf'], 2)} {dados['area']}",
                'Redução': f"{formatar_numero_brasileiro(dados['reducao_percentual'], 1)}%",
                'Impacto Rendimento': f"{formatar_numero_brasileiro(dados.get('reducao_rendimento', dados.get('aumento_rendimento', 0)), 1)}%"
            })
        
        df_comparacao = pd.DataFrame(comparacao_data)
        st.dataframe(df_comparacao)

if __name__ == "__main__":
    main()
