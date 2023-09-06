import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px


def translate_feedback(feedback_df, column_name):
    feedback_df["translated"] = "-"  # Add a new column "translated" and initialize all rows with "-"

    for i, feedback in enumerate(feedback_df[column_name]):
        try:
            translation = GoogleTranslator(source='auto', target='en').translate(feedback)
            feedback_df.loc[i, "translated"] = translation  # Store the translation in the "translated" column
        except Exception as e:
            feedback_df.loc[i, "translated"] = "-"  # Store "-" in the "translated" column if an error occurs

    feedback_df = feedback_df[feedback_df["translated"] != "-"]  # Remove "-" rows
    return feedback_df

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode the DataFrame as base64
    href = f'<a href="data:file/csv;base64,{b64}" download="translated_feedback.csv">Download CSV file</a>'
    return href

def topics_over_time(topic_model, dataframe, training_column):
    timestamps = list(dataframe.day.values)
    feedback_list = list(dataframe[training_column])
    topics_over_time = topic_model.topics_over_time(feedback_list, timestamps, global_tuning=True, evolution_tuning=True)
    f = topic_model.visualize_topics_over_time(topics_over_time, custom_labels=True)
    f.update_layout(width=800,height=500)
    return f
    
def area_over_time(topic_model, df, training_column, datetime_column):
    df['Topic'] = topic_model.get_document_info(df[training_column])["Name"].values

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month

    # Group the data by year, month, and topic
    grouped = df.groupby(['year', 'month', 'Topic'])[training_column].count().reset_index()

    # Normalize the document counts by the total document count for each month and topic
    grouped['total_count'] = grouped.groupby(['year', 'month'])[training_column].transform('sum')
    grouped['document_pct'] = grouped[training_column] / grouped['total_count'] * 100

    # Pivot the data to create a table with months as rows, topics as columns, and document percentages as values
    pivoted = pd.pivot_table(grouped, index=['year', 'month'], columns='Topic', values='document_pct', fill_value=0)
    pivoted = pivoted.reset_index()

    # Melt the data to create a long format with separate rows for each topic
    melted = pd.melt(pivoted, id_vars=['year', 'month'], var_name='Topic', value_name='document_pct')

    # Create the interactive plot using Plotly Express
    fig = px.area(melted, x='month', y='document_pct', color='Topic', facet_col='year', facet_col_wrap=3,
                  title='Distribution of Documents by Topic and Month (Relative to 100%)',
                  labels={'month': 'Month', 'document_pct': 'Document Percentage', 'Topic': 'Topic', 'year': 'Year'},
                  hover_data={'month': False, 'document_pct': ':.2f'})

    return fig

# Sidebar configuration
st.sidebar.title("Translation and Analysis App")
tab = st.sidebar.selectbox("Select Tab", ("Translate", "Analyse Feedback"))


if tab == "Translate":
    st.title("Translate Feedback")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)
    
    if file is not None:
        file.seek(0)
        feedback_df = pd.read_csv(file, low_memory=False, on_bad_lines='skip', engine='c') if file.name.endswith(".csv") else pd.read_excel(file)
        st.write('**Data Head:**')
        st.write(feedback_df.head())
        column_name = st.selectbox("Select Column", feedback_df.columns)
        feedback_df = feedback_df.dropna(subset=[column_name])
        feedback_df = feedback_df.reset_index(drop=True)
        if st.button("Translate"):
            translated_df = translate_feedback(feedback_df, column_name)
            csv = convert_df(translated_df)

            st.write('**Translated Data Head:**')
            st.write(translated_df.head())

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='translated_data.csv',
                mime='text/csv',
            )

            

elif tab == "Analyse Feedback":
    # Analyse Feedback tab code
    st.title("Analyse Feedback")

    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        df = pd.read_csv(file, on_bad_lines='skip') if file.name.endswith(".csv") else pd.read_excel(file)
        st.write('**Data Head:**')
        st.write(df.head())
        column_names = df.columns.tolist()

        datetime_column = st.selectbox("Select Datetime Column", column_names + ["None"])
        feedback_column = st.selectbox("Select Feedback Column", column_names)

        model_select = st.selectbox(
            "Select model to train:",
            [
                'all-mpnet-base-v2',
                'all-distilroberta-v1',
                'distiluse-base-multilingual-cased-v2',
                'multi-qa-mpnet-base-dot-v1',
                'multi-qa-distilbert-cos-v1',
                'paraphrase-multilingual-mpnet-base-v2'
            ]
        )

        if st.button("Train Model"):
            if model_select is not None:
                new_df = df.copy()
                if datetime_column != "None":
                    new_df[datetime_column] = pd.to_datetime(new_df[datetime_column])
                sentence_model = SentenceTransformer(model_select)

                vectorizer_model = CountVectorizer(stop_words="english")

                # Initialize a BERTopic model with the SentenceTransformer embeddings
                my_model = BERTopic(
                    language="en",
                    calculate_probabilities=True,
                    verbose=True,
                    n_gram_range=(1, 3),
                    embedding_model=sentence_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics = 15
                )

                # Preprocess the data by replacing missing values with empty strings
                new_df[feedback_column] = new_df[feedback_column].fillna('')
                new_df.reset_index(inplace = True,drop = True)

                # Fit the BERTopic model on the dataframe
                my_model.fit(new_df[feedback_column])
                st.success("Model trained successfully")

                # Store the trained model in session state
                st.session_state.trained_model = my_model
                st.session_state.new_df = new_df
                st.session_state.feedback_colomn = feedback_column
                st.session_state.datetime_column = datetime_column

        if "trained_model" in st.session_state:
            trained_model = st.session_state.trained_model
            new_df = st.session_state.new_df
            new_feedback_column = st.session_state.feedback_colomn
            

            visualization_options = [
                "Visualize documents",
                "Topic Hierarchy",
                "Barchart",
                "Topics over time",
                "Representative docs per topic"
            ]
            selected_visualization = st.selectbox("Select Visualization", visualization_options)

            if selected_visualization == "Barchart":
                umap_fig = trained_model.visualize_barchart(n_words=5)
                st.plotly_chart(umap_fig)
            elif selected_visualization == "Visualize documents":
                viz_doc = trained_model.visualize_documents(new_df[new_feedback_column])
                st.plotly_chart(viz_doc)
            elif selected_visualization == "Topic Hierarchy":
                tsne_fig = trained_model.visualize_hierarchy(top_n_topics=20)
                st.plotly_chart(tsne_fig)
            elif selected_visualization == "Topics over time":
                time_fig = area_over_time(trained_model, new_df, new_feedback_column, datetime_column)
                st.plotly_chart(time_fig)
            elif selected_visualization == "Representative docs per topic":    
                st.write(trained_model.get_representative_docs())

            result = pd.merge(new_df[feedback_column], 
                              trained_model.get_document_info(new_df[feedback_column]),
                              left_on=feedback_column,
                              right_on='Document',
                              how = 'left'
                              )

            feedback_and_docs = convert_df(result)
            st.download_button(
                label="Download documents and topics",
                data=feedback_and_docs,
                file_name='document_info.csv',
                mime='text/csv',
            )
