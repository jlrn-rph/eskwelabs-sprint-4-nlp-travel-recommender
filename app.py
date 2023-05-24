import pandas as pd
import streamlit as st
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Lakbay Mindanao: A Data-Driven Journey Through the Hidden Gems of the South")

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def load_data():
    data = pd.read_csv("./data/travel_packages_ph.csv", encoding="ISO-8859-1")
    data = data.drop(data.columns[0], axis=1)
    return data

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''

def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    return similarity

def recommend_destination(user_destination, user_activity, user_budget, user_duration):
    # Load the data
    data = load_data()

    # Convert user inputs to title case
    user_destination = user_destination.title()
    user_activity = user_activity.title()

    # Filter data based on the user's destination, duration, and budget
    user_filtered_data = data[(data['location'] == user_destination) & (data['duration'] <= user_duration) & (data['price'] <= user_budget)]

    if user_filtered_data.empty: # No similar results found
        return pd.DataFrame()

    # Filter data based on the Mindanao region and exact activity match
    mindanao_exact_data = data[(data['region'] == 'Mindanao') & (data['activity'] == user_activity) & (data['duration'] <= user_duration) & (data['price'] <= user_budget)]

    if not mindanao_exact_data.empty:
        # Exact activity match found in Mindanao, return the top 3 recommendations
        mindanao_exact_data_sorted = mindanao_exact_data.sort_values(by='duration', ascending=False)
        return mindanao_exact_data_sorted.head(3)

    # Preprocess the user's destination description
    user_filtered_data['preprocessed_description'] = user_filtered_data['description_clean'].apply(preprocess_text)

    # Preprocess the Mindanao destination descriptions
    mindanao_data = data[data['region'] == 'Mindanao']
    mindanao_data['preprocessed_description'] = mindanao_data['description_clean'].apply(preprocess_text)

    # Compute TF-IDF vectors for the Mindanao destination descriptions
    tfidf = TfidfVectorizer()
    mindanao_tfidf_matrix = tfidf.fit_transform(mindanao_data['preprocessed_description'])

    # Compute TF-IDF vectors for the user's destination descriptions
    user_tfidf_matrix = tfidf.transform(user_filtered_data['preprocessed_description'])

    # Compute cosine similarity between user's destination and Mindanao destinations
    similarity_matrix = cosine_similarity(user_tfidf_matrix, mindanao_tfidf_matrix)

    if similarity_matrix.size == 0:
        return pd.DataFrame()  # No similarity found, return empty DataFrame

    # Get the index of the most similar Mindanao destinations
    destination_index = similarity_matrix[0].argsort()[::-1]

    # Get the cosine similarity values
    cosine_similarities = similarity_matrix[0, destination_index]

    # Calculate activity similarity for user's input
    user_activity = nlp(user_activity)
    mindanao_data['activity_similarity'] = mindanao_data['activity'].apply(lambda x: user_activity.similarity(nlp(str(x))) if isinstance(x, str) else 0.0)

    # Calculate average similarity for each destination
    recommended_destinations = mindanao_data.iloc[destination_index]
    recommended_destinations['cosine_similarity'] = cosine_similarities
    recommended_destinations['average_similarity'] = (recommended_destinations['activity_similarity'] + recommended_destinations['cosine_similarity']) / 2.0

    # Filter recommendations based on duration
    filtered_destinations = recommended_destinations[(recommended_destinations['duration'] <= user_duration) & (recommended_destinations['price'] <= user_budget)]

    # Sort recommendations based on average similarity
    sorted_destinations = filtered_destinations.sort_values(by='average_similarity', ascending=False)

    # Get the top 3 recommendations
    top_recommendations = sorted_destinations.head(3)

    return top_recommendations

def recommendation_page():
    st.image("./assets/title.png")
    st.title("Recommendation System for Mindanao Tourist Trip Packages")
    st.markdown(
        """
        This recommender system suggests travel packages in Mindanao based on the user's preferred activity, duration, budget, and previous destination. It uses a content-based approach to match their preferences with the characteristics of the destinations.

        The system analyzes the textual descriptions of destinations using TF-IDF vectorization to capture important terms and computes cosine similarity to measure text similarity.

        If there is an exact match between the preferred activity and a destination in Mindanao, those destinations will be recommended. Otherwise, the system will find destinations in Mindanao that are similar to the preferred destination based on their textual descriptions. It also considers the similarity between your preferred activity and the activities associated with the destinations using Spacy's similarity measure.

        To get personalized recommendations, enter the previous destination, preferred activity, duration, and budget, and the system will suggest the best matches for the tourist packages in Mindanao.
        """
    )

    data = load_data()

    data_luzvis = data[data['region'] != 'Mindanao']

    user_destination = st.selectbox("Where is your travel inspiration?", data_luzvis['location'].unique())
    user_activity = st.text_input("What activity do you want to do?")
    user_budget = st.number_input("How much budget do you have?", min_value=5.0, max_value=160000.0)
    user_duration = st.number_input("How long do you want the trip to last?", min_value=0.0, max_value=120.0)

    if st.button("Tara!"):
        try:
            recommendations = recommend_destination(user_destination, user_activity, user_budget, user_duration)
            # st.dataframe(recommendations)

            if recommendations.empty:
                st.warning("🌵🌄❌\nOops! No hidden gems found for your preferences. But don't worry, there are plenty of other amazing places to discover. Keep exploring!")
            else:
                st.subheader("💎 Here's your hidden gem!")
                
                for index, recommendation in recommendations.iterrows():
                    title = recommendation['title']
                    location = recommendation['location']
                    activity = recommendation['activity']
                    price = recommendation['price']
                    duration = recommendation['duration']
                    url = recommendation['url']

                    st.subheader(f"{title}")
                    st.markdown(f"**📍 Location:**\n{location}\n"
                                f"**🏃‍♀️ Activity:**\n{activity}\n"
                                f"**💸 Price:**\n{price}\n"
                                f"**🕒 Duration:**\n{str(duration)}\n")
                    st.markdown(f"🔗 \n[Get this package now!]({url})\n")
        except AttributeError:
            st.error("❌ Oops! Something went wrong on our end. We're currently checking it out. We apologize for the inconvenience and appreciate your understanding. Please try another input.")

# Meet the team
def the_team():
    st.title("The Team")
    st.markdown(
        """
        We are the team **chatgpt4**! We are a group of individuals from diverse backgrounds who came together as part of the Eskwelabs Data Science Cohort 11. In our fourth sprint, we collaborated to create a data-driven presentation on NLP-based recommender engine entitled **Lakbay Mindanao: A Data-Driven Journey Through the Hidden Gems of the South**. 

        The project uses data scraped from Klook and TripAdvisor, which is preprocessed by tokenization, lemmatization, and other NLP-based preprocessing techniques. It is wrangled and analyzed using Python Pandas, exploratory data analysis using Matplotlib, and similarity techniques using TF-IDF and spaCy similarity . Recommender engines were also utilized to provide personalized recommendation for potential travel packages in Mindanao.
        """
    )
    st.header("Members")
    st.subheader("[Kurt Chester Laconico](https://www.linkedin.com/in/kurt-chester-laconico-462967180/)")

    st.subheader("[Patrick Jonathan Atienza](https://www.linkedin.com/in/patrick-jonathan-atienza-5002b160/)")

    st.subheader("[Reynaly Shen Javier](https://www.linkedin.com/in/reynaly-shen-javier/)")

    st.subheader("[Zheena Halagao](https://www.linkedin.com/in/zheena-halagao-6b9486107/)")

    st.subheader("[Justin Louise Neypes](https://www.linkedin.com/in/jlrnrph/)")

    st.subheader("Mentor: Karen")

# Define the main menu
list_of_pages = [
    "Travel Package Recommender",
    "The Team"
]

st.sidebar.title(':scroll: Main Menu')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "Travel Package Recommender":
    recommendation_page()

elif selection == "The Team":
    the_team()
