import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def weight_loss_predict(age, exercise, water, sleep, calories):
    """
    Predict weight loss based on user inputs using linear regression
    
    Args:
        age: User's age
        exercise: Exercise hours per week
        water: Daily water intake in liters
        sleep: Average sleep hours
        calories: Daily calorie intake
    
    Returns:
        Predicted weight loss in kg for 1 month
    """
    try:
        # Load the dataset
        df = pd.read_csv("fitness_health_tracking.csv")
        
        # Prepare the features and target
        features = ['Age', 'Exercise_Hours_per_Week', 'Daily_Water_Intake_L', 
                   'Average_Sleep_Hours', 'Daily_Calories_Intake']
        target = 'Weight_Loss_1_Month_kg'
        
        # Check if required columns exist
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns in dataset: {missing_cols}")
            return None
        
        # Create and train the model
        reg = LinearRegression()
        X = df[features]
        y = df[target]
        reg.fit(X, y)
        
        # Make prediction using DataFrame to maintain feature names
        user_data = pd.DataFrame([[age, exercise, water, sleep, calories]], 
                                columns=features)
        prediction = reg.predict(user_data)
        
        return prediction[0]
        
    except FileNotFoundError:
        print("Error: 'fitness_health_tracking.csv' file not found.")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_user_input():
    """
    Collect user information with input validation
    
    Returns:
        Tuple of (age, exercise, water, sleep, calories)
    """
    try:
        print("=== Weight Loss Prediction System ===")
        print("Please enter your information:")
        
        age = int(input("Enter your age: "))
        if age <= 0 or age > 120:
            print("Warning: Age should be between 1 and 120")
        
        exercise = float(input("Enter your exercise hours per week: "))
        if exercise < 0 or exercise > 168:  # 168 hours in a week
            print("Warning: Exercise hours should be between 0 and 168")
        
        water = float(input("Enter your daily water intake in liters: "))
        if water < 0 or water > 20:
            print("Warning: Water intake should be between 0 and 20 liters")
        
        sleep = float(input("Enter your average sleep hours per day: "))
        if sleep < 0 or sleep > 24:
            print("Warning: Sleep hours should be between 0 and 24")
        
        calories = int(input("Enter your daily calorie intake: "))
        if calories < 0 or calories > 10000:
            print("Warning: Calorie intake seems unusual")
        
        return age, exercise, water, sleep, calories
        
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return None

def main():
    """Main function to run the weight loss prediction system"""
    user_data = get_user_input()
    
    if user_data is None:
        print("Failed to get user input. Please try again.")
        return
    
    age, exercise, water, sleep, calories = user_data
    
    print("\n=== Your Input ===")
    print(f"Age: {age}")
    print(f"Exercise Hours per Week: {exercise}")
    print(f"Daily Water Intake: {water}L")
    print(f"Average Sleep Hours: {sleep}")
    print(f"Daily Calorie Intake: {calories}")
    
    print("\n=== Processing Prediction ===")
    prediction = weight_loss_predict(age, exercise, water, sleep, calories)
    
    if prediction is not None:
        print(f"\n=== Prediction Result ===")
        print(f"Predicted weight loss in 1 month: {prediction:.2f} kg")
        
        if prediction > 0:
            print("Great! The model predicts you will lose weight.")
        elif prediction < 0:
            print("The model predicts you might gain weight.")
        else:
            print("The model predicts your weight will remain stable.")
            
        print("\nNote: This is a prediction based on historical data and may not reflect individual variations.")
    else:
        print("Failed to make prediction. Please check your data file and try again.")

if __name__ == "__main__":
    main()