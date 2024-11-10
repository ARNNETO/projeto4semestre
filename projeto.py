# Importa bibliotecas
import streamlit as st
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import matplotlib.pyplot as plt
import numpy as np

# Função para baixar e preparar dados com cache
@st.cache_data
def download_and_prepare_data():
    # Inicia a API
    api = KaggleApi()
    api.authenticate()

    # Diretório onde o dataset será baixado e descompactado
    dataset_dir = './climate-change-global-temperature-data/datasets'

    # Baixa os arquivos
    api.dataset_download_files('sachinsarkar/climate-change-global-temperature-data', path=dataset_dir, unzip=True)

    # Coleta caminho dos arquivos
    file_path_country = os.path.join(dataset_dir, 'GlobalLandTemperaturesByCountry.csv')
    file_path_state = os.path.join(dataset_dir, 'GlobalLandTemperaturesByState.csv')

    # Cria os dataframes
    df_country = pd.read_csv(file_path_country, 
                             dtype={
                                 'AverageTemperature': 'float64',
                                 'AverageTemperatureUncertainty': 'float64',
                                 'Country': 'object'
                             },
                             parse_dates=['dt'])
    df_state = pd.read_csv(file_path_state, 
                           dtype={
                               'AverageTemperature': 'float64',
                               'AverageTemperatureUncertainty': 'float64',
                               'State': 'object',
                               'Country': 'object'
                           },
                           parse_dates=['dt'])
    return df_country, df_state

# Carrega dados com cache
df_country, df_state = download_and_prepare_data()

# Configura a barra lateral
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Selecione a página", ["Home", "Explore a Temperatura"])

# Página "Home"
if page == "Home":
    # Adiciona o texto descritivo sobre aquecimento global
    st.write("""
        # Monitoramento da Temperatura Global
        
        Monitorar e analisar o aumento das temperaturas globais para identificar tendências e possíveis impactos climáticos. 
        Este projeto é essencial para fornecer dados precisos que suportem políticas ambientais e ações para mitigar as mudanças climáticas.
        
        A alta temperatura global é um fenômeno crescente que tem despertado preocupação em todo o mundo. Esse aquecimento anômalo da superfície terrestre é resultado de uma combinação de fatores naturais e, principalmente, da atividade humana, que intensificou o efeito estufa. A emissão de gases como o dióxido de carbono (CO₂), metano (CH₄) e óxidos de nitrogênio (NOx) contribuem para a retenção de calor na atmosfera, levando a um aumento das temperaturas médias globais.

        Esse aquecimento tem consequências graves para o equilíbrio ambiental e para a vida na Terra. Ele causa o derretimento das calotas polares, eleva o nível do mar, e torna os eventos climáticos extremos mais frequentes, como ondas de calor, secas, incêndios florestais e tempestades intensas. Além disso, o aumento das temperaturas afeta a biodiversidade, forçando muitas espécies a migrar ou até desaparecer, alterando ecossistemas inteiros.

        As temperaturas recordes registradas em várias regiões do mundo nos últimos anos são um sinal claro da urgência de adotar medidas para mitigar o aquecimento global. Entre essas medidas estão a redução das emissões de gases de efeito estufa, o incentivo ao uso de energias renováveis, e a preservação das florestas, que são essenciais para a absorção de CO₂. A conscientização e a ação coletiva são fundamentais para que possamos limitar o aumento das temperaturas e proteger o planeta para as futuras gerações.
    """)
    
    st.image("/home/neto/Imagens/AltaTemperaturaGlobal.jpg", caption="Aquecimento global e seus impactos", use_container_width=True)

# Página "Explore a Temperatura"
elif page == "Explore a Temperatura":
    st.write("### Exploração de Temperatura")

    # Upload de arquivos
    uploaded_file = st.file_uploader("Faça o upload de um arquivo CSV para adicionar mais informações", type=["csv"])
    
    if uploaded_file:
        # Carregar arquivo CSV
        additional_data = pd.read_csv(uploaded_file, parse_dates=['dt'])
        st.write("Dados carregados com sucesso:")
        st.write(additional_data.head())
        
        # Concatena os dados
        if 'Country' in additional_data.columns:
            df_country = pd.concat([df_country, additional_data], ignore_index=True)
            st.write("Dados adicionais foram mesclados com os dados por país.")
        elif 'State' in additional_data.columns:
            df_state = pd.concat([df_state, additional_data], ignore_index=True)
            st.write("Dados adicionais foram mesclados com os dados por estado.")

    # Estado de sessão para os valores dos filtros
    if "start_year_country" not in st.session_state:
        st.session_state.start_year_country = int(df_country['dt'].dt.year.min())
    if "end_year_country" not in st.session_state:
        st.session_state.end_year_country = int(df_country['dt'].dt.year.max())
    if "country" not in st.session_state:
        st.session_state.country = df_country['Country'].unique()[0]

    # Filtro para df_country
    st.write("#### Filtrar Temperatura por País")
    start_year_country = st.slider(
        "Ano Início", min_value=int(df_country['dt'].dt.year.min()), 
        max_value=int(df_country['dt'].dt.year.max()), 
        value=st.session_state.start_year_country, 
        key="start_year_country"
    )
    end_year_country = st.slider(
        "Ano Fim", min_value=int(df_country['dt'].dt.year.min()), 
        max_value=int(df_country['dt'].dt.year.max()), 
        value=st.session_state.end_year_country, 
        key="end_year_country"
    )
    country = st.selectbox(
        "Selecione o País", df_country['Country'].unique(), 
        index=list(df_country['Country'].unique()).index(st.session_state.country), 
        key="country"
    )

    # Filtraos dados
    df_filtered_country = df_country[
        (df_country['dt'].dt.year >= start_year_country) &
        (df_country['dt'].dt.year <= end_year_country) &
        (df_country['Country'] == country)
    ]

    # Verifica se há dados após o filtro antes de plotar o gráfico
    if not df_filtered_country.empty:
        df_country_avg = df_filtered_country.resample('Y', on='dt')['AverageTemperature'].mean().dropna()
        
        fig, ax = plt.subplots()
        ax.plot(df_country_avg.index.year, df_country_avg.values, label="Média Anual")

        z = np.polyfit(df_country_avg.index.year, df_country_avg.values, 1)
        p = np.poly1d(z)
        
        ax.plot(df_country_avg.index.year, p(df_country_avg.index.year), "r--", label="Linha de Tendência")
        ax.set_title(f'Média Anual de Temperatura em {country}')
        ax.set_xlabel("Ano")
        ax.set_ylabel("Média de Temperatura (°C)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Não há dados disponíveis para os filtros selecionados.")

    # Serviço de download de dados filtrados
    if not df_filtered_country.empty:
        csv_data = df_filtered_country.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar dados filtrados por país",
            data=csv_data,
            file_name=f"temperatura_{country}_{start_year_country}_{end_year_country}.csv",
            mime="text/csv"
        )

    # Estado de sessão para os filtros de estado
    if "start_year_state" not in st.session_state:
        st.session_state.start_year_state = int(df_state['dt'].dt.year.min())
    if "end_year_state" not in st.session_state:
        st.session_state.end_year_state = int(df_state['dt'].dt.year.max())
    if "country_state" not in st.session_state:
        st.session_state.country_state = df_state['Country'].unique()[0]
    if "state" not in st.session_state:
        st.session_state.state = df_state[df_state['Country'] == st.session_state.country_state]['State'].unique()[0]

    # Filtro para df_state
    st.write("#### Filtrar Temperatura por Estado e País")
    start_year_state = st.slider(
        "Ano Início", min_value=int(df_state['dt'].dt.year.min()), 
        max_value=int(df_state['dt'].dt.year.max()), 
        value=st.session_state.start_year_state, key="start_year_state"
    )
    end_year_state = st.slider(
        "Ano Fim", min_value=int(df_state['dt'].dt.year.min()), 
        max_value=int(df_state['dt'].dt.year.max()), 
        value=st.session_state.end_year_state, key="end_year_state"
    )
    country_state = st.selectbox(
        "Selecione o País (Estado)", df_state['Country'].unique(), 
        index=list(df_state['Country'].unique()).index(st.session_state.country_state), 
        key="country_state"
    )
    state = st.selectbox(
        "Selecione o Estado", 
        df_state[df_state['Country'] == country_state]['State'].unique(), 
        index=list(df_state[df_state['Country'] == country_state]['State'].unique()).index(st.session_state.state), 
        key="state"
    )

    # Filtra os dados
    df_filtered_state = df_state[
        (df_state['dt'].dt.year >= start_year_state) &
        (df_state['dt'].dt.year <= end_year_state) &
        (df_state['Country'] == country_state) &
        (df_state['State'] == state)
    ]

    # Verifica se há dados após o filtro antes de plotar o gráfico
    if not df_filtered_state.empty:
        df_state_avg = df_filtered_state.resample('Y', on='dt')['AverageTemperature'].mean().dropna()
        
        fig, ax = plt.subplots()
        ax.plot(df_state_avg.index.year, df_state_avg.values, label="Média Anual")

        z = np.polyfit(df_state_avg.index.year, df_state_avg.values, 1)
        p = np.poly1d(z)
        
        ax.plot(df_state_avg.index.year, p(df_state_avg.index.year), "r--", label="Linha de Tendência")
        ax.set_title(f'Média Anual de Temperatura em {state}, {country_state}')
        ax.set_xlabel("Ano")
        ax.set_ylabel("Média de Temperatura (°C)")
        ax.legend()
        
        # Exibe o gráfico
        st.pyplot(fig)
        plt.close(fig)

        # Serviço de download de dados filtrados para o segundo gráfico
        csv_data_state = df_filtered_state.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar dados filtrados por estado",
            data=csv_data_state,
            file_name=f"temperatura_{state}_{country_state}_{start_year_state}_{end_year_state}.csv",
            mime="text/csv"
        )
    else:
        st.write("Não há dados disponíveis para os filtros selecionados.")
