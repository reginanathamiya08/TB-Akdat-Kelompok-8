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

            # Visualization of hero attributes (bar plot)
            fig, ax = plt.subplots(figsize=(10, 6))
            hero_attributes_cleaned.plot(kind='bar', ax=ax, color='skyblue')
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=10)
            ax.set_title(f"Attributes of {hero_name} (After Preprocessing)")
            ax.set_ylabel('Normalized Value')
            ax.set_xlabel('Attributes')
            st.pyplot(fig)

    # Add New Hero Data
    st.subheader("Add New Hero Data")

    # Input fields for new hero data
    new_hero_name = st.text_input("Hero Name")
    new_hero_role = st.selectbox("Hero Role", ['Fighter', 'Mage', 'Marksman', 'Tank', 'Support', 'Assassin'])

    new_hero_data = {}
    for feature in features:
        # Append the feature name with a unique identifier to ensure uniqueness
        new_hero_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, value=0.5, step=0.01, key=f"{feature}_new")

    # Initialize data_cleaned as the original data if it has not been cleaned yet
    if data_cleaned is None:
        data_cleaned = data.copy()

    if st.button("Add Hero"):
        if new_hero_name:
            # Append the new hero data to the existing dataset
            new_hero_data['hero_name'] = new_hero_name
            new_hero_data['role'] = new_hero_role
            new_hero_df = pd.DataFrame([new_hero_data])

            # Append to the original data
            data_cleaned = pd.concat([data_cleaned, new_hero_df], ignore_index=True)

            st.success(f"Hero {new_hero_name} added successfully!")

            # Display updated dataset
            st.subheader("Updated Dataset")
            st.dataframe(data_cleaned)

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

            # Input for each attribute
            custom_input = {}
            for feature in model_features:
                # Append the feature name with a unique identifier to ensure uniqueness
                custom_input[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, value=0.5, step=0.01, key=f"{feature}_predict")

            # Create a DataFrame for the input values
            input_data = pd.DataFrame([custom_input])

            # Predict role
            predicted_role_encoded = model.predict(input_data)[0]
            predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]
            
            st.write(f"Predicted Role: **{predicted_role}**")
else:
    st.write("Please upload a CSV file to proceed.")
