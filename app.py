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

def introduction():
    st.image("./assets/title.png")
    st.markdown(
        """
        One of the best things about being a Filipino is living in a country full of stunning beaches, majestic mountains, and many more. While you've been to or heard a lot about places like Boracay and Batanes from the country's northern parts, you might not know a lot in Mindanao due to misconceptions about safety.

        But it's just as beautiful so today, let us take you on a data-driven journey through the hidden gems of the South.
        """
    )
    st.image("./assets/eda-attention.png")
    st.write("Only 3% of trip listings in Klook and TripAdvisor are in Mindanao. It can gain more attention through the help of travel websites looking for more trip listings from the region.")

    st.image("./assets/eda-highlight.png")
    st.write("Only five tourist spots in Mindanao are listed on these websites but it has so much more. There are places like Mt. Apo, the country's highest peak and Maria Cristina Falls that supply Mindanao with electricity, which can be highlighted more.")

    st.image("./assets/intro-investment.png")
    st.write("There is no Mindanao attraction among the top 10 highest-rated tourist spots in the Philippines. It could be a cycle of Mindanao not being known so locals have less incentive to develop the place for tourism which leads again to not being known for tourism. Thus, aside from better marketing, Mindanao needs to invest more in accommodating tourists.")

    st.image("./assets/eda-variety.png")
    st.write("One way to do that is to provide more varied activities for visitors because currently, Mindanao only has three of the top activities in tourist trips in the country.")

    st.image("./assets/intro-surf.png")
    st.write("It could actually capitalize on offering unique activities like surfing for which Siargao is very popular.")

    st.image("./assets/eda-afford.png")
    st.write("It also has the opportunity to sell its affordability given the many budget Filipino travelers out there. It takes just half of what you have to pay to travel in Luzon but with equally breathtaking views and memorable experiences.")

    st.image("./assets/intro-luzvis.png")
    st.write("Compared to Luzon and Visayas, Mindanao has encountered security challenges due to the presence of several rebel groups. The island has experienced intermittent occurrences of bombings, kidnappings, and war. This gave Mindanao the impression of a chaotic island and led to the discouragement of traveling in the region.")

    st.image("./assets/intro-dot.png")
    st.markdown(
        """
        Last September 2022,the Philippine Tour Operators Association announced Mindanao as a secure and appealing tourist destination. 

        In order to boost tourism in the region, the Department of Tourism and Cebu Pacific launched a new flight route linking Clark and Cagayan de Oro airport, making it easier for tourists to access the area.

        Another project of DOT is the **Colors of Mindanao**, a promotional campaign which aims to show the world that the island has more to offer in terms of tourist attractions, culinary dishes, and culture than most people realize. 

        These plans aim to dispel the negative associations linked to past incidents, safety concerns, conflicts and ultimately promote a positive image of Mindanao.
        """)
    
    st.subheader("Objectives")
    st.image("./assets/objectives.png")
    st.write("Our objective is to promote tourism in the Philippines, specifically in Mindanao. The researchers will accomplish this by creating a recommender engine where users can input information such as their budget, desired activity, duration, and inspiration place and be given recommendations of available tourist trips or packages from the dataset.")

    st.image("./assets/recommender-different.png")
    st.markdown(
        """
         Most travel sites, such as Klook or Tripadviser, while they do have specific categories for places and activities, they only ask the user for a location and give out general suggestions. 

        What makes our recommender different is that it offers a more curated experience by also asking their budget, duration of trip, and what kind of activity they want to experience. Then it will show available tourist packages in Mindanao.
        """
        )

def methodology():
    st.title("Methodology")
    st.image("./assets/methodology.png")
    st.write("The diagram shows a high level view of the methodology. The group did web scraping from two travel websites namely Klook and TripAdvisor. The data is pre-processed through cleaning and standardization. It is analyzed through EDA to display top destinations and activities as well as several other relevant information. Along with the user input, the pre-processed data will go through several filters, and semantic and textual similarity scores. The engine will give the top recommended travel packages in Mindanao based on your preferences. ")

    st.image("./assets/scraping.png")
    st.write("The data was obtained from Klook using Selenium and TripAdvisor using BeautifulSoup to get 1005 raw, clean travel package entries. Features captured were location, title, description, price, rating, review count, duration of activity and the link to the package. The region and activity group features are mapped during pre-processing. The activity feature for Klook is raw, and the TripAdvisors’ activity feature was gathered using text processing techniques.")

    st.subheader("Data Preprocessing")
    st.image("./assets/data-preprocessing.png")
    st.write("NLP-related pre-processing methods were also applied to the data set. Tokenization and Part of Speech (POS) were used to capture the activity from the package descriptions. Lemmatization, sentence separation, removal of punctuations and stop words were applied to the descriptions. There was also mapping of the locations with the corresponding region, and the activity with the corresponding activity group like tours, water activities, recreational and others. The format of the duration, review counts, and price features were standardized. After which, dropping and filling out of null values were applied.")

    st.subheader("How does the recommender engine work?")
    st.image("./assets/method.png")
    st.markdown(
        """
        1. **User Inputs**: The system takes input from the user, including the destination they have been to, their preferred activity, duration, and budget.

        2. **Filtering**: The system filters the dataset based on the user's destination, duration, and budget. It selects destinations that match the user's destination, duration, and have a price within the user's budget.

        3. **Exact Activity Match**: The system checks if there are any destinations in Mindanao that exactly match the user's preferred activity. If a match is found, those destinations are recommended.

        4. **TF-IDF Vectorization**: If there is no exact activity match, the system preprocesses the destination descriptions and computes TF-IDF vectors for both the user's destination descriptions and the Mindanao destination descriptions. TF-IDF captures the importance of terms in the descriptions and represents them numerically.

        5. **Cosine Similarity**: The system calculates the cosine similarity between the TF-IDF vectors of the user's destination and the Mindanao destinations. This measures the similarity between the textual contents of the descriptions.

        6. **Activity Similarity**: The system also calculates the similarity between the user's preferred activity and the activities associated with the Mindanao destinations using Spacy's similarity measure. This measures the semantic similarity between the user's activity and the activities of the destinations.

        7. **Average Similarity**: The system combines the cosine similarity and the activity similarity by taking their average for each destination.

        8. **Ranking and Recommendation**: The system ranks the Mindanao destinations based on the average similarity and recommends the top three destinations with the highest average similarity.
        """
    )

    data = load_data()
    with st.expander("Data: Philippine Travel Packages"):
        st.dataframe(data)
        st.caption("*Source: Klook and TripAdvisor*")

def recommendation():
    st.title("Recommendations")
    st.write("This recommender will be very useful but there are still improvements to be made, namely...")
    st.image('./assets/recommendation.png')
    st.image('./assets/recommendation-1.png')
    st.image('./assets/recommendation-2.png')
    st.image('./assets/message.png')

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
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True)
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
    user_activity = st.text_input("What activity do you want to do?", placeholder="e.g. Day Trips, Hiking, etc.")
    user_budget = st.number_input("How much budget do you have?", min_value=5.0, max_value=160000.0)
    user_duration = st.number_input("How long do you want the trip to last (in hours)?", min_value=0.0, max_value=120.0)

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
    st.markdown(
        """
        * Conducted data cleaning and preprocessing on the Klook dataset, employing NLP-related techniques such as tokenization, Part of Speech (POS) tagging, lemmatization, sentence separation, punctuation removal, and stop word elimination. These processes were applied to ensure standardization of the data.
        """
    )

    st.subheader("[Patrick Jonathan Atienza](https://www.linkedin.com/in/patrick-jonathan-atienza-5002b160/)")
    st.markdown(
        """
        * Utilized BeautifulSoup to conduct web scraping on TripAdvisor, extracting various details about travel packages. This includes information such as the package's location, title, description, price, rating, review count, duration of activity, and the corresponding link to the package.
        """
    )

    st.subheader("[Reynaly Shen Javier](https://www.linkedin.com/in/reynaly-shen-javier/)")
    st.markdown(
        """
        * Led the charge in the overall analysis and presentation of results, ensuring that our findings were clearly and persuasively communicated to our audience.
        * Produced informative data visualizations through exploratory data analysis showcasing insights from the scraped travel packages and tourist trips across the Philippines, especially in Mindanao. 
        """
    )

    st.subheader("[Zheena Halagao](https://www.linkedin.com/in/zheena-halagao-6b9486107/)")
    st.markdown(
        """
        * Performed data cleaning and preprocessing using NLP-related techniques such tokenization, Part of Speech (POS), lemmatization, sentence separation, removal of punctuations and stop words, etc. to standardize the TripAdvisor data.
        * Conducted comprehensive research on the reasons of the majority of trips in Luzon and Visayas; and debunking misconceptions through DOT and PHILTOA campaign.
        """
    )

    st.subheader("[Justin Louise Neypes](https://www.linkedin.com/in/jlrnrph/)")
    st.markdown(
        """
        * Spearheaded the design and deployment of the Sprint 4 project on Streamlit, showcasing the findings to others.
        * Performed web scraping using Selenium to extract travel package information from Klook such as the location, title, description, price, rating, review count, duration of activity and the link to the package.
        * Developed a recommender engine to give personalized travel packages recommendations for travelers based on their destination inspiration, activity, budget, and duration.
        """
    )

    st.subheader("Mentor: Karen Bioy")

# Define the main menu
list_of_pages = [
    "About the Project",
    "Methodology",
    "Travel Package Recommender",
    "Recommendations",
    "The Team"
]

st.sidebar.title(':scroll: Main Menu')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "About the Project":
    introduction()

elif selection == "Methodology":
    methodology()

elif selection == "Travel Package Recommender":
    recommendation_page()

elif selection == "Recommendations":
    recommendation()

elif selection == "The Team":
    the_team()
