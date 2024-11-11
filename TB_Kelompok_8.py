import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.title("MLBB Hero Role Prediction and Analysis App")
st.write("This app provides visualizations for hero attributes and predicts hero roles based on custom inputs.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Check if 'role' exists in the dataset
    if 'role' in data.columns:
        # Hero selection for visualization
        st.subheader("Select a Hero to Visualize Attributes")
        hero_name = st.selectbox("Choose a Hero", data['hero_name'].unique())

        # Filter data for the selected hero
        hero_data = data[data['hero_name'] == hero_name].iloc[0]
        hero_attributes = {
            'defense_overall': hero_data['defense_overall'],
            'offense_overall': hero_data['offense_overall'],
            'skill_effect_overall': hero_data['skill_effect_overall'],
            'difficulty_overall': hero_data['difficulty_overall'],
            'movement_spd': hero_data['movement_spd'],
            'magic_defense': hero_data['magic_defense'],
            'mana': hero_data['mana'],
            'hp_regen': hero_data['hp_regen'],
            'physical_atk': hero_data['physical_atk'],
            'physical_defense': hero_data['physical_defense'],
            'hp': hero_data['hp'],
            'attack_speed': hero_data['attack_speed'],
            'mana_regen': hero_data['mana_regen'],
            'win_rate': hero_data['win_rate'],
            'pick_rate': hero_data['pick_rate'],
            'ban_rate': hero_data['ban_rate']
        }

        # Display the hero's attributes as a bar chart
        st.subheader(f"Attributes of {hero_name}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(hero_attributes.keys(), hero_attributes.values(), color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel("Attributes")
        plt.ylabel("Value")
        plt.title(f"Attribute Analysis for {hero_name}")
        st.pyplot(fig)

        # Encode the target 'role' column
        label_encoder = LabelEncoder()
        data['role_encoded'] = label_encoder.fit_transform(data['role'])

        # Define features and target
        features = [
            'defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall',
            'movement_spd', 'magic_defense', 'mana', 'hp_regen'
        ]
        
        X = data[features]
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
                                    movement_spd, magic_defense, mana, hp_regen]], columns=features)

        # Predict role
        predicted_role_encoded = model.predict(input_data)[0]
        predicted_role = label_encoder.inverse_transform([predicted_role_encoded])[0]
        
        st.write(f"Predicted Role: **{predicted_role}**")

    else:
        st.error("Column 'role' not found in the dataset. Please ensure the file has a 'role' column.")
else:
    st.write("Please upload a CSV file to proceed.")
