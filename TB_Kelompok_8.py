import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Title and Description
st.title("MLBB Hero Role Prediction and Analysis App")
st.write("This app provides preprocessing, outlier detection, visualizations, and hero role predictions.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Display the full dataset as a table
    st.subheader("Full Dataset")
    st.dataframe(data)

    # Display columns to help debug if 'role' column is missing
    st.write("Columns in the dataset:")
    st.write(data.columns)  # Display the columns to check if 'role' is present

    # Hero selection for visualization
    st.subheader("Select a Hero to Visualize Attributes")
    hero_name = st.selectbox("Choose a Hero", data['hero_name'].unique())

    # Display hero attributes before preprocessing
    if hero_name:
        hero_data_original = data[data['hero_name'] == hero_name].iloc[0]
        hero_attributes_original = hero_data_original[
            ['defense_overall', 'offense_overall', 'skill_effect_overall',
             'difficulty_overall', 'movement_spd', 'magic_defense', 'mana',
             'hp_regen', 'physical_atk', 'physical_defense', 'hp',
             'attack_speed', 'mana_regen', 'win_rate', 'pick_rate', 'ban_rate']
        ]

        st.subheader(f"Attributes of {hero_name} (Before Preprocessing)")
        st.table(hero_attributes_original.reset_index().rename(columns={'index': 'Attribute', 0: 'Value'}))

    # Initialize data_cleaned variable
    data_cleaned = data.copy()  # Initialize this variable to avoid errors

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
        features = [
            'defense_overall', 'offense_overall', 'skill_effect_overall',
            'difficulty_overall', 'movement_spd', 'magic_defense', 'mana',
            'hp_regen', 'physical_atk', 'physical_defense', 'hp',
            'attack_speed', 'mana_regen', 'win_rate', 'pick_rate', 'ban_rate'
        ]
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

        # Visualization section: Graphs after preprocessing
        st.subheader("Visualizations After Preprocessing")

        # Bar chart for feature comparison
        st.write("### Feature Comparison (Bar Chart)")
        hero_features = features[:6]  # Select a subset of features to display
        hero_values = hero_data_cleaned[hero_features]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(hero_features, hero_values, color='skyblue')
        plt.xticks(rotation=45)
        plt.xlabel("Attributes")
        plt.ylabel("Normalized Value")
        plt.title(f"Attributes Comparison for {hero_name}")
        st.pyplot(fig)

        # Histogram for distribution of features (For overall dataset)
        st.write("### Distribution of Features (Histograms)")
        for feature in features[:6]:  # Show histograms for the first 6 features
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data_cleaned[feature], bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature.replace('_', ' ').capitalize()}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Scatter plots for relationships between features (example: 'physical_atk' vs 'hp')
        st.write("### Scatter Plot for Relationship Between Features")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data_cleaned, x='physical_atk', y='hp', hue='role', palette='viridis', ax=ax)
        ax.set_title("Physical Attack vs HP (Color-coded by Role)")
        ax.set_xlabel("Physical Attack")
        ax.set_ylabel("HP")
        st.pyplot(fig)

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        heatmap_fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data_cleaned[features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(heatmap_fig)

    # Check if 'role' exists in the dataset for prediction
    if 'role' not in data.columns:
        st.error("The 'role' column is missing from the dataset. Please ensure the dataset includes this column.")
    else:
        # Encode the target 'role' column if preprocessing was done
        if 'role_encoded' not in data_cleaned.columns:
            st.warning("Preprocessing has not been performed. Please click 'Run Outlier Detection and Preprocessing' first.")
        else:
            # Encode the 'role' column into numeric values for prediction
            label_encoder = LabelEncoder()
            data_cleaned['role_encoded'] = label_encoder.fit_transform(data_cleaned['role'])

            # Define features and target for the prediction model
            model_features = [
                'defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall',
                'movement_spd', 'magic_defense', 'mana', 'hp_regen'
            ]

            X = data_cleaned[model_features]
            y = data_cleaned['role_encoded']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the decision tree model
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
