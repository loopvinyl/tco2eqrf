# =============================================================================
# FUNÇÕES DE COTAÇÃO DE ARROZ E TRIGO
# =============================================================================

def obter_cotacao_arroz():
    """Obtém a cotação em tempo real do arroz em casca"""
    try:
        # API do Cepea (Centro de Estudos Avançados em Economia Aplicada)
        url = "https://www.cepea.esalq.usp.br/br/indicador/arroz.aspx"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tentar encontrar o preço mais recente
        preco = None
        # Procura por padrões comuns
        padroes = [
            r'R\$\s*[\d\.]+,\d{2}',
            r'[\d\.]+,\d{2}\s*\/\s*sc',
            r'valor.*?[\d\.]+,\d{2}'
        ]
        
        import re
        texto = soup.get_text()
        for padrao in padroes:
            matches = re.findall(padrao, texto, re.IGNORECASE)
            if matches:
                # Extrair número do primeiro match
                match = matches[0]
                num = re.search(r'[\d\.]+,\d{2}', match)
                if num:
                    preco_texto = num.group(0).replace('.', '').replace(',', '.')
                    preco = float(preco_texto)
                    # Converter de sc (60kg) para tonelada se necessário
                    if 'sc' in match.lower():
                        preco = preco * (1000/60)  # Converter para R$/ton
                    break
        
        if preco and 800 < preco < 3000:  # Faixa razoável para arroz
            return preco, True, "Cepea"
        return 1500, False, "Referência"  # Fallback
        
    except Exception as e:
        return 1500, False, f"Erro: {str(e)}"

def obter_cotacao_trigo():
    """Obtém a cotação em tempo real do trigo"""
    try:
        # API do Cepea para trigo
        url = "https://www.cepea.esalq.usp.br/br/indicador/trigo.aspx"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        preco = None
        import re
        texto = soup.get_text()
        
        # Procurar por valores de trigo
        matches = re.findall(r'R\$\s*([\d\.]+,\d{2})', texto)
        if matches:
            preco_texto = matches[0].replace('.', '').replace(',', '.')
            preco = float(preco_texto)
            # Verificar se é por saca (60kg) ou tonelada
            if 'sc' in texto.lower() or 'saca' in texto.lower():
                preco = preco * (1000/60)  # Converter para R$/ton
        
        if preco and 1000 < preco < 2500:
            return preco, True, "Cepea"
        return 1200, False, "Referência"
        
    except Exception as e:
        return 1200, False, f"Erro: {str(e)}"

# =============================================================================
# ATUALIZAR DADOS DOS ARTIGOS
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
        'sistema': 'Monocultura',
        'rendimento_arroz_base': 7.0,  # ton/ha (mantido para compatibilidade)
        'rendimento_trigo_base': 0,    # Não aplicável
        'preco_arroz_base': 1500,
        'preco_trigo_base': 0,
        'unidade_rendimento': 'ton/ha (arroz)',
        'tem_duas_culturas': False
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
        'sistema': 'Rotação',
        'rendimento_arroz_base': 6.0,  # ton/ha (arroz)
        'rendimento_trigo_base': 4.0,  # ton/ha (trigo)
        'preco_arroz_base': 1500,
        'preco_trigo_base': 1200,
        'unidade_rendimento': 'ton/ha (arroz + trigo)',
        'tem_duas_culturas': True
    }
}

# =============================================================================
# ATUALIZAR FUNÇÃO CALCULAR_RENDIMENTO
# =============================================================================

def calcular_rendimento_duas_culturas(tipo, rend_arroz, rend_trigo, preco_arroz, preco_trigo, area_ha, estudo):
    """
    Calcula rendimento ajustado para sistema com duas culturas
    """
    dados = DADOS_ARTIGOS[estudo]
    
    if tipo.lower() == 'convencional':
        fator_ajuste = 1.0
    else:  # CRF
        fator_ajuste = 1 + (dados['aumento_rendimento'] / 100)
    
    # Aplicar ajuste a ambas as culturas
    rend_arroz_ajustado = rend_arroz * fator_ajuste
    rend_trigo_ajustado = rend_trigo * fator_ajuste
    
    # Calcular totais
    rend_total_arroz = rend_arroz_ajustado * area_ha
    rend_total_trigo = rend_trigo_ajustado * area_ha
    rend_total_ton = rend_total_arroz + rend_total_trigo
    rend_total_ha = rend_arroz_ajustado + rend_trigo_ajustado
    
    # Calcular valor total
    valor_total = (rend_total_arroz * preco_arroz) + (rend_total_trigo * preco_trigo)
    valor_ha = (rend_arroz_ajustado * preco_arroz) + (rend_trigo_ajustado * preco_trigo)
    
    return {
        'total_ton': rend_total_ton,
        'total_ha': rend_total_ha,
        'total_valor': valor_total,
        'valor_ha': valor_ha,
        'arroz_ton': rend_total_arroz,
        'trigo_ton': rend_total_trigo,
        'arroz_ha': rend_arroz_ajustado,
        'trigo_ha': rend_trigo_ajustado
    }

def calcular_rendimento(tipo, rendimento_base, area_ha, estudo, rend_trigo=None, preco_arroz=None, preco_trigo=None):
    """
    Função wrapper que chama a função apropriada baseada no estudo
    """
    dados = DADOS_ARTIGOS[estudo]
    
    if dados['tem_duas_culturas']:
        # Usar função para duas culturas
        return calcular_rendimento_duas_culturas(
            tipo, 
            rendimento_base,  # Aqui é rendimento do arroz
            rend_trigo or dados['rendimento_trigo_base'],
            preco_arroz or dados['preco_arroz_base'],
            preco_trigo or dados['preco_trigo_base'],
            area_ha, 
            estudo
        )
    else:
        # Sistema monocultura (mantém lógica original)
        if tipo.lower() == 'convencional':
            fator_ajuste = 1.0
        else:  # CRF
            fator_ajuste = 1 + (dados['reducao_rendimento'] / 100)
        
        rendimento_ajustado_ha = rendimento_base * fator_ajuste
        rendimento_total = rendimento_ajustado_ha * area_ha
        
        return {
            'total_ton': rendimento_total,
            'total_ha': rendimento_ajustado_ha,
            'total_valor': rendimento_total * (preco_arroz or dados['preco_arroz_base']),
            'valor_ha': rendimento_ajustado_ha * (preco_arroz or dados['preco_arroz_base'])
        }

# =============================================================================
# ATUALIZAR A SIDEBAR NO main()
# =============================================================================

# No main(), dentro da sidebar, substituir a seção de parâmetros da cultura:

def main():
    # ... (código anterior mantido)
    
    with st.sidebar:
        # ... (código anterior mantido)
        
        st.subheader("📍 Parâmetros da Cultura")
        
        # Sistema monocultura (Ji et al.)
        if estudo_selecionado == 'ji_et_al':
            rendimento_arroz = st.slider(
                f"Rendimento do Arroz (ton/ha)",
                min_value=float(max(1.0, dados_estudo['rendimento_arroz_base'] * 0.5)),
                max_value=float(dados_estudo['rendimento_arroz_base'] * 2.0),
                value=float(dados_estudo['rendimento_arroz_base']),
                step=0.5,
                help="Rendimento médio do arroz com fertilizante convencional"
            )
            
            # Botão para buscar cotação do arroz
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Atualizar Preço do Arroz", key="atualizar_arroz"):
                    st.session_state.cotacao_arroz_atualizada = True
            
            if st.session_state.get('cotacao_arroz_atualizada', False):
                with st.spinner("Buscando cotação do arroz..."):
                    preco_arroz, sucesso, fonte = obter_cotacao_arroz()
                    st.session_state.preco_arroz = preco_arroz
                    st.session_state.fonte_arroz = fonte
                    st.session_state.cotacao_arroz_atualizada = False
                    st.rerun()
            
            if 'preco_arroz' not in st.session_state:
                st.session_state.preco_arroz = dados_estudo['preco_arroz_base']
                st.session_state.fonte_arroz = "Referência"
            
            preco_arroz = st.slider(
                f"Preço do Arroz (R$/ton)",
                min_value=int(dados_estudo['preco_arroz_base'] * 0.5),
                max_value=int(dados_estudo['preco_arroz_base'] * 2.0),
                value=int(st.session_state.preco_arroz),
                step=50,
                help=f"Fonte: {st.session_state.fonte_arroz}"
            )
            
            rendimento_trigo = 0
            preco_trigo = 0
        
        # Sistema rotação (Shakoor et al.)
        else:  # shakoor_et_al
            col1, col2 = st.columns(2)
            with col1:
                rendimento_arroz = st.slider(
                    "Rendimento do Arroz (ton/ha)",
                    min_value=float(max(1.0, dados_estudo['rendimento_arroz_base'] * 0.5)),
                    max_value=float(dados_estudo['rendimento_arroz_base'] * 2.0),
                    value=float(dados_estudo['rendimento_arroz_base']),
                    step=0.5
                )
            
            with col2:
                rendimento_trigo = st.slider(
                    "Rendimento do Trigo (ton/ha)",
                    min_value=float(max(1.0, dados_estudo['rendimento_trigo_base'] * 0.5)),
                    max_value=float(dados_estudo['rendimento_trigo_base'] * 2.0),
                    value=float(dados_estudo['rendimento_trigo_base']),
                    step=0.5
                )
            
            # Cotações de arroz e trigo
            st.write("**Cotações em tempo real:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Arroz", key="atualizar_arroz_rotacao"):
                    st.session_state.cotacao_arroz_atualizada = True
            
            with col2:
                if st.button("🔄 Trigo", key="atualizar_trigo"):
                    st.session_state.cotacao_trigo_atualizada = True
            
            # Buscar cotações se necessário
            if st.session_state.get('cotacao_arroz_atualizada', False):
                with st.spinner("Buscando cotação do arroz..."):
                    preco_arroz_temp, sucesso, fonte = obter_cotacao_arroz()
                    st.session_state.preco_arroz = preco_arroz_temp
                    st.session_state.fonte_arroz = fonte
                    st.session_state.cotacao_arroz_atualizada = False
                    st.rerun()
            
            if st.session_state.get('cotacao_trigo_atualizada', False):
                with st.spinner("Buscando cotação do trigo..."):
                    preco_trigo_temp, sucesso, fonte = obter_cotacao_trigo()
                    st.session_state.preco_trigo = preco_trigo_temp
                    st.session_state.fonte_trigo = fonte
                    st.session_state.cotacao_trigo_atualizada = False
                    st.rerun()
            
            # Inicializar preços se necessário
            if 'preco_arroz' not in st.session_state:
                st.session_state.preco_arroz = dados_estudo['preco_arroz_base']
                st.session_state.fonte_arroz = "Referência"
            
            if 'preco_trigo' not in st.session_state:
                st.session_state.preco_trigo = dados_estudo['preco_trigo_base']
                st.session_state.fonte_trigo = "Referência"
            
            col1, col2 = st.columns(2)
            with col1:
                preco_arroz = st.slider(
                    "Preço do Arroz (R$/ton)",
                    min_value=int(dados_estudo['preco_arroz_base'] * 0.5),
                    max_value=int(dados_estudo['preco_arroz_base'] * 2.0),
                    value=int(st.session_state.preco_arroz),
                    step=50,
                    help=f"Fonte: {st.session_state.fonte_arroz}"
                )
            
            with col2:
                preco_trigo = st.slider(
                    "Preço do Trigo (R$/ton)",
                    min_value=int(dados_estudo['preco_trigo_base'] * 0.5),
                    max_value=int(dados_estudo['preco_trigo_base'] * 2.0),
                    value=int(st.session_state.preco_trigo),
                    step=50,
                    help=f"Fonte: {st.session_state.fonte_trigo}"
                )
        
        # ... (restante do código da sidebar mantido)

# =============================================================================
# ATUALIZAR OS CÁLCULOS NA SEÇÃO DE SIMULAÇÃO
# =============================================================================

# No trecho onde executa a simulação, atualizar:

if st.session_state.executar_simulacao:
    with st.spinner('Executando simulação...'):
        # ... (código anterior mantido)
        
        # Calcular rendimentos (atualizado para lidar com duas culturas)
        rendimento_conv = calcular_rendimento(
            'convencional', 
            rendimento_arroz, 
            area_total, 
            estudo_selecionado,
            rend_trigo if estudo_selecionado == 'shakoor_et_al' else None,
            preco_arroz,
            preco_trigo if estudo_selecionado == 'shakoor_et_al' else None
        )
        
        rendimento_crf = calcular_rendimento(
            'crf', 
            rendimento_arroz, 
            area_total, 
            estudo_selecionado,
            rend_trigo if estudo_selecionado == 'shakoor_et_al' else None,
            preco_arroz,
            preco_trigo if estudo_selecionado == 'shakoor_et_al' else None
        )
        
        # Nas métricas, ajustar para mostrar dados separados se for rotação
        if estudo_selecionado == 'shakoor_et_al':
            st.subheader("🌾 Rendimentos por Cultura")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Arroz Convencional",
                    f"{formatar_br(rendimento_conv['arroz_ha'])} ton/ha",
                    f"R$ {formatar_br(rendimento_conv['arroz_ha'] * preco_arroz)}/ha"
                )
            
            with col2:
                st.metric(
                    "Trigo Convencional",
                    f"{formatar_br(rendimento_conv['trigo_ha'])} ton/ha",
                    f"R$ {formatar_br(rendimento_conv['trigo_ha'] * preco_trigo)}/ha"
                )
            
            with col3:
                st.metric(
                    "Arroz CRF",
                    f"{formatar_br(rendimento_crf['arroz_ha'])} ton/ha",
                    f"R$ {formatar_br(rendimento_crf['arroz_ha'] * preco_arroz)}/ha"
                )
            
            with col4:
                st.metric(
                    "Trigo CRF",
                    f"{formatar_br(rendimento_crf['trigo_ha'])} ton/ha",
                    f"R$ {formatar_br(rendimento_crf['trigo_ha'] * preco_trigo)}/ha"
                )
        
        # Nas métricas gerais, usar os totais
        st.metric(
            f"Impacto no Rendimento ({dados_estudo['cultura']})",
            f"R$ {formatar_br(rendimento_crf['total_valor'])}",
            f"R$ {formatar_br(rendimento_crf['total_valor'] - rendimento_conv['total_valor'])}",
            help=f"Valor total da produção com CRF vs Convencional"
        )
        
        # ... (restante do código mantido)
