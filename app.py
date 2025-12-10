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

# Preços do carbono (referência)
PRECO_CARBONO_EUR = 85.50  # €/tCO₂eq (valor de referência)
PRECO_CARBONO_REAL = 85.50 * 5.5  # Conversão para R$ (€1 = R$5,50)

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
        
        # Configurações avançadas
        with st.expander("🔧 Parâmetros Avançados"):
            preco_carbono_eur = st.number_input(
                "Preço do Carbono (€/tCO₂eq)",
                min_value=10.0,
                max_value=200.0,
                value=85.5,
                step=5.0
            )
            
            taxa_cambio = st.number_input(
                "Taxa Câmbio (€ → R$)",
                min_value=4.0,
                max_value=7.0,
                value=5.5,
                step=0.1
            )
            
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
            
            # Calcular receita do carbono
            receita_carbono_real, receita_carbono_eur = calcular_receita_carbono(
                reducao_tco2eq_total,
                preco_carbono_eur,
                taxa_cambio
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
                'preco_carbono': preco_carbono_eur,
                'taxa_cambio': taxa_cambio,
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
                'preco_carbono': preco_carbono_eur,
                'taxa_cambio': taxa_cambio,
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
                    "Redução de Emissões",
                    f"{reducao_tco2eq_total:,.0f} tCO₂eq",
                    delta=f"{dados_estudo['reducao_percentual']}%"
                )
            
            with col2:
                st.metric(
                    "Receita Carbono Potencial",
                    f"R$ {receita_carbono_real:,.0f}",
                    f"€ {receita_carbono_eur:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Custo Adicional CRF",
                    f"R$ {custo_crf - custo_convencional:,.0f}",
                    f"{((custo_crf_ha/custo_conv_ha)-1)*100:.1f}% mais caro"
                )
            
            with col4:
                if estudo_selecionado == 'ji_et_al':
                    delta_rend = f"{dados_estudo['reducao_rendimento']}%"
                else:
                    delta_rend = f"+{dados_estudo['aumento_rendimento']}%"
                
                st.metric(
                    "Impacto no Rendimento",
                    f"{rendimento_crf:,.0f} ton",
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
            
            # Gráfico 2: Distribuição Monte Carlo (VPL)
            axes[1].hist(resultados_mc['vpl'], bins=30, edgecolor='black', alpha=0.7)
            axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Ponto de Equilíbrio')
            axes[1].axvline(x=np.mean(resultados_mc['vpl']), color='g', linestyle='-', 
                           linewidth=2, label=f'Média: R$ {np.mean(resultados_mc["vpl"]):,.0f}')
            axes[1].set_xlabel('VPL (R$/ha)')
            axes[1].set_ylabel('Frequência')
            axes[1].set_title('Distribuição do VPL (Monte Carlo)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
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
                    f"{np.mean(resultados_mc['viabilidade']) * 100:.1f}%",
                    help="Percentual de simulações onde VPL > 0"
                )
                
                st.metric(
                    "VPL Médio",
                    f"R$ {np.mean(resultados_mc['vpl']):,.0f}/ha",
                    help="Valor Presente Líquido médio por hectare"
                )
                
                st.metric(
                    "Intervalo de Confiança 95%",
                    f"[R$ {np.percentile(resultados_mc['vpl'], 2.5):,.0f}, R$ {np.percentile(resultados_mc['vpl'], 97.5):,.0f}]",
                    help="Intervalo de confiança do VPL"
                )
            
            with col2:
                st.write("#### Viabilidade Base")
                st.metric(
                    "VPL do Projeto",
                    f"R$ {resultados_viabilidade['vpl'] * area_total:,.0f}",
                    f"R$ {resultados_viabilidade['vpl']:,.0f}/ha"
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
                        preco_minimo_eur = preco_minimo_ha / taxa_cambio
                        
                        st.metric(
                            "Preço Mínimo do Carbono para Viabilidade",
                            f"€ {preco_minimo_eur:,.0f}/tCO₂eq",
                            f"R$ {preco_minimo_ha:,.0f}/tCO₂eq"
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
            
            # Criar cenários
            cenarios = [
                {'nome': 'Cenário Atual', 'preco_carbono': 85.5, 'taxa_cambio': 5.5},
                {'nome': 'Mercado em Expansão', 'preco_carbono': 120, 'taxa_cambio': 5.5},
                {'nome': 'Alta do Carbono', 'preco_carbono': 150, 'taxa_cambio': 5.5},
                {'nome': 'Mercado Regulado', 'preco_carbono': 200, 'taxa_cambio': 5.5}
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
                    'Preço Carbono (€)': cenario['preco_carbono'],
                    'VPL Total (R$)': vpl_cenario,
                    'VPL/ha (R$)': vpl_cenario / area_total,
                    'Viável': 'SIM' if vpl_cenario > 0 else 'NÃO'
                })
            
            df_cenarios = pd.DataFrame(resultados_cenarios)
            st.dataframe(df_cenarios.style.highlight_max(subset=['VPL Total (R$)'], color='lightgreen')
                         .highlight_min(subset=['VPL Total (R$)'], color='lightcoral'))
            
            # =================================================================
            # 9. CONCLUSÕES E RECOMENDAÇÕES
            # =================================================================
            st.subheader("🎯 Conclusões e Recomendações")
            
            vpl_ha = resultados_viabilidade['vpl']
            probabilidade_viabilidade = np.mean(resultados_mc['viabilidade']) * 100
            
            if vpl_ha > 0:
                st.success(f"""
                **✅ PROJETO VIÁVEL**
                
                - **VPL positivo:** R$ {vpl_ha * area_total:,.0f} (R$ {vpl_ha:,.0f}/ha)
                - **Probabilidade de sucesso:** {probabilidade_viabilidade:.1f}%
                - **Payback:** {resultados_viabilidade['payback']} anos
                
                **Recomendações:**
                1. Implementar projeto piloto em área reduzida
                2. Buscar certificação VCS ou Gold Standard
                3. Negociar contratos de venda antecipada de créditos
                4. Aproveitar ganhos de produtividade (se aplicável)
                """)
            else:
                st.warning(f"""
                **⚠️ PROJETO NÃO VIÁVEL NO CENÁRIO ATUAL**
                
                - **VPL negativo:** R$ {vpl_ha * area_total:,.0f} (R$ {vpl_ha:,.0f}/ha)
                - **Probabilidade de viabilidade:** {probabilidade_viabilidade:.1f}%
                - **Fator limitante:** Custo adicional do CRF
                
                **Estratégias para viabilizar:**
                1. Buscar subsídios governamentais para transição
                2. Negociar desconto com fornecedores de CRF
                3. Esperar aumento no preço do carbono (viável a partir de € {preco_minimo_eur if 'preco_minimo_eur' in locals() else 'N/A':,.0f}/tCO₂eq)
                4. Focar no aumento de produtividade como principal benefício
                5. Considerar combinação CRF + ureia para reduzir custos
                """)
            
            # Adicionar insights específicos por estudo
            with st.expander("📚 Insights Específicos por Estudo"):
                if estudo_selecionado == 'ji_et_al':
                    st.info("""
                    **Ji et al. (2013) - Sistema Arroz:**
                    - CRF reduz emissões em 14.5%, mas reduz rendimento em 5%
                    - Timing da aeração (MSA) é crítico: MSA em D30 otimiza redução
                    - Necessário compensar perda de rendimento com valor agregado ou carbono
                    """)
                else:
                    st.info("""
                    **Shakoor et al. (2018) - Sistema Arroz-Trigo:**
                    - CRF reduz emissões em 26.5% e aumenta rendimento em 3%
                    - Sistema de rotação otimiza benefícios
                    - Viabilidade mais provável devido ao duplo benefício
                    """)
            
            # =================================================================
            # 10. DOWNLOAD DOS RESULTADOS
            # =================================================================
            st.subheader("💾 Download dos Resultados")
            
            # Preparar dados para exportação
            dados_exportacao = {
                'Parâmetros de Entrada': {
                    'Estudo Base': dados_estudo['nome'],
                    'Área Total (ha)': area_total,
                    'Anos Simulação': anos_simulacao,
                    'Rendimento Base (ton/ha)': rendimento_base,
                    'Preço Produto (R$/ton)': preco_produto,
                    'Preço Carbono (€/tCO₂eq)': preco_carbono_eur,
                    'Taxa Câmbio (€→R$)': taxa_cambio,
                    'Taxa Desconto (%)': taxa_desconto * 100
                },
                'Resultados Principais': {
                    'Redução Emissões (tCO₂eq)': reducao_tco2eq_total,
                    'Receita Carbono (R$)': receita_carbono_real,
                    'Custo Convencional (R$)': custo_convencional,
                    'Custo CRF (R$)': custo_crf,
                    'Custo Adicional (R$)': custo_crf - custo_convencional,
                    'Rendimento Convencional (ton)': rendimento_conv,
                    'Rendimento CRF (ton)': rendimento_crf,
                    'VPL Total (R$)': resultados_viabilidade['vpl'] * area_total,
                    'VPL/ha (R$)': resultados_viabilidade['vpl'],
                    'Payback (anos)': resultados_viabilidade['payback'],
                    'Probabilidade Viabilidade (%)': probabilidade_viabilidade
                }
            }
            
            df_exportar = pd.DataFrame([
                {'Categoria': 'Entrada', 'Parâmetro': k, 'Valor': v} 
                for k, v in dados_exportacao['Parâmetros de Entrada'].items()
            ] + [
                {'Categoria': 'Resultado', 'Parâmetro': k, 'Valor': v} 
                for k, v in dados_exportacao['Resultados Principais'].items()
            ])
            
            # Converter para CSV
            csv = df_exportar.to_csv(index=False)
            
            st.download_button(
                label="📥 Baixar Resultados (CSV)",
                data=csv,
                file_name=f"resultados_fertilizantes_{estudo_selecionado}.csv",
                mime="text/csv"
            )
    
    else:
        # Tela inicial
        st.info("""
        ### 💡 Como usar este simulador:
        
        1. **Selecione o estudo base** na barra lateral (Ji et al. 2013 ou Shakoor et al. 2018)
        2. **Configure os parâmetros** da sua operação (área, rendimento, preços)
        3. **Clique em "Executar Simulação Completa"**
        4. **Analise os resultados** de viabilidade econômica e ambiental
        
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
                'Emissão Convencional': f"{dados['emissao_convencional']} {dados['area']}",
                'Emissão CRF': f"{dados['emissao_crf']} {dados['area']}",
                'Redução': f"{dados['reducao_percentual']}%",
                'Impacto Rendimento': f"{dados.get('reducao_rendimento', dados.get('aumento_rendimento', 0))}%"
            })
        
        df_comparacao = pd.DataFrame(comparacao_data)
        st.dataframe(df_comparacao)

if __name__ == "__main__":
    main()
