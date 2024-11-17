import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Title and Description
st.title("MLBB Hero Role Prediction and Analysis App")
st.write("This app provides visualizations for hero attributes, preprocessing options, and predicts hero roles based on custom inputs.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

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

    # Filter data for the selected hero before preprocessing
    hero_data_original = data[data['hero_name'] == hero_name].iloc[0]
    hero_attributes_original = hero_data_original[features]

    # Display the hero's attributes as a table before preprocessing
    st.subheader(f"Attributes of {hero_name} (Before Preprocessing)")
    st.table(hero_attributes_original.reset_index().rename(columns={'index': 'Attribute', 0: 'Value'}))

    # Button to run outlier detection and handling
    if st.button("Run Outlier Detection and Handling"):
        # Function to detect and handle outliers using IQR
        def handle_outliers(data, features):
            for feature in features:
                Q1 = data[feature].quantile(0.25)  # First quartile
                Q3 = data[feature].quantile(0.75)  # Third quartile
                IQR = Q3 - Q1                      # Interquartile range
                lower_bound = Q1 - 1.5 * IQR       # Lower bound
                upper_bound = Q3 + 1.5 * IQR       # Upper bound

                # Clip the outliers to within the bounds
                data[feature] = data[feature].clip(lower=lower_bound, upper=upper_bound)
            return data

        # Apply the function to handle outliers
        data_cleaned = handle_outliers(data.copy(), features)

        st.success("Outlier detection and handling completed. Outliers have been clipped to the valid range.")

        # Display the full dataset after handling outliers
        st.subheader("Dataset After Handling Outliers")
        st.dataframe(data_cleaned)

        # Filter data for the selected hero after outlier handling
        hero_data_cleaned = data_cleaned[data_cleaned['hero_name'] == hero_name].iloc[0]
        hero_attributes_cleaned = hero_data_cleaned[features]

        # Display the hero's attributes as a table after handling outliers
        st.subheader(f"Attributes of {hero_name} (After Outlier Handling)")
        st.table(hero_attributes_cleaned.reset_index().rename(columns={'index': 'Attribute', 0: 'Value'}))

    # Check if 'role' exists in the dataset for prediction
    if 'role' in data.columns:
        # Encode the target 'role' column
        label_encoder = LabelEncoder()
        data['role_encoded'] = label_encoder.fit_transform(data['role'])

        # Define features and target
        model_features = [
            'defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall',
            'movement_spd', 'magic_defense', 'mana', 'hp_regen'
        ]
        
        X = data[model_features]
        y = data['role_encoded']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Prediction based on custom inputs
        st.subheader("Predict Role Based on Custom Attributes")
        
        defense_overall = st.number_input("Defense Overall", min_value=0, max_value=10, value=5)
        offense_overall = st.number_input("Offense Overall", min_value=0, max_value=10, value=5)
        skill_effect_overall = st.number_input("Skill Effect Overall", min_value=0, max_value=10, value=5)
        difficulty_overall = st.number_input("Difficulty Overall", min_value=0, max_value=10, value=5)
        movement_spd = st.number_input("Movement Speed", min_value=0, max_value=300, value=250)
        magic_defense = st.number_input("Magic Defense", min_value=0, max_value=100, value=10)
        mana = st.number_input("Mana", min_value=0, max_value=1000, value=400)
        hp_regen = st.number_input("HP Regen", min_value=0, max_value=100, value=30)

        # Create a DataFrame for the input values
        input_data = pd.DataFrame([[defense_overall, offense_overall, skill_effect_overall, difficulty_overall,
                                    movement_spd, magic_defense, mana, hp_regen]], columns=model_features)

        # Predict role
        predicted_role_encoded = model.predict(input_data)[0]
        predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]
        
        st.write(f"Predicted Role: **{predicted_role}**")
    else:
        st.error("Column 'role' not found in the dataset. Please ensure the file has a 'role' column.")
else:
    st.write("Please upload a CSV file to proceed.")
