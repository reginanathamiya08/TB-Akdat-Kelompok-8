import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Title and Description
st.title("MLBB Hero Role Prediction and Analysis App")
st.write("This app provides preprocessing, outlier detection, visualizations, and hero role predictions.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Initialize a variable to keep track of whether preprocessing was done
data_cleaned = None

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Display the full dataset as a table
    st.subheader("Full Dataset")
    st.dataframe(data)

    # Define numeric columns for preprocessing
    features = [
        'defense_overall', 'offense_overall', 'skill_effect_overall',
        'difficulty_overall', 'movement_spd', 'magic_defense', 'mana',
        'hp_regen', 'physical_atk', 'physical_defense', 'hp',
        'attack_speed', 'mana_regen', 'win_rate', 'pick_rate', 'ban_rate'
    ]

    # Hero selection for visualization
    st.subheader("Select a Hero to Visualize Attributes")
    hero_name = st.selectbox("Choose a Hero", data['hero_name'].unique())

    # Display hero attributes before preprocessing
    if hero_name:
        hero_data_original = data[data['hero_name'] == hero_name].iloc[0]
        hero_attributes_original = hero_data_original[features]

        st.subheader(f"Attributes of {hero_name} (Before Preprocessing)")
        st.table(hero_attributes_original.reset_index().rename(columns={'index': 'Attribute', 0: 'Value'}))

    # Button to run preprocessing
    if st.button("Run Outlier Detection and Preprocessing"):
        # Function to detect and handle outliers using IQR
        def handle_outliers(data, features):
            for feature in features:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[feature] = data[feature].clip(lower=lower_bound, upper=upper_bound)
            return data

        # Apply the function to handle outliers
        data_cleaned = handle_outliers(data.copy(), features)

        # Normalize numeric features for consistency
        scaler = MinMaxScaler()
        data_cleaned[features] = scaler.fit_transform(data_cleaned[features])

        st.success("Outlier detection and preprocessing completed!")

        # Display cleaned dataset
        st.subheader("Dataset After Preprocessing")
        st.dataframe(data_cleaned)

        # Display hero attributes after preprocessing
        if hero_name:
            hero_data_cleaned = data_cleaned[data_cleaned['hero_name'] == hero_name].iloc[0]
            hero_attributes_cleaned = hero_data_cleaned[features]

            st.subheader(f"Attributes of {hero_name} (After Preprocessing)")
            st.table(hero_attributes_cleaned.reset_index().rename(columns={'index': 'Attribute', 0: 'Normalized Value'}))

        # Visualizations (Histogram)
        st.subheader("Visualizations (Histogram)")

        # Histogram for 'win_rate'
        st.write("### Histogram of Win Rate")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data_cleaned['win_rate'], kde=True, ax=ax, color='blue', bins=20)
        st.pyplot(fig)

        # Histogram for 'pick_rate'
        st.write("### Histogram of Pick Rate")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data_cleaned['pick_rate'], kde=True, ax=ax, color='green', bins=20)
        st.pyplot(fig)

        # Histogram for 'ban_rate'
        st.write("### Histogram of Ban Rate")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data_cleaned['ban_rate'], kde=True, ax=ax, color='red', bins=20)
        st.pyplot(fig)

    # Check if 'role' exists in the dataset for prediction
    if 'role' not in data.columns:
        st.error("Column 'role' not found in the dataset. Please ensure the file has a 'role' column.")
        st.write("Kolom yang ada dalam dataset:", data.columns)
    else:
        # Proceed only if preprocessing was done and data_cleaned is not None
        if data_cleaned is not None:
            # Encode the target 'role' column
            label_encoder = LabelEncoder()
            data_cleaned['role_encoded'] = label_encoder.fit_transform(data_cleaned['role'])

            # Define features and target
            model_features = [
                'defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall',
                'movement_spd', 'magic_defense', 'mana', 'hp_regen'
            ]
            
            X = data_cleaned[model_features]
            y = data_cleaned['role_encoded']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the model
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Prediction based on custom inputs
            st.subheader("Predict Role Based on Custom Attributes")
            
            custom_input = {}
            for feature in model_features:
                custom_input[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, value=0.5)

            # Create a DataFrame for the input values
            input_data = pd.DataFrame([custom_input])

            # Predict role
            predicted_role_encoded = model.predict(input_data)[0]
            predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]
            
            st.write(f"Predicted Role: **{predicted_role}**")
else:
    st.write("Please upload a CSV file to proceed.")
