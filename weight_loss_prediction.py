import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import csv

def weight_loss_predict(age, exercise, water, sleep, calories):
    try:
        df = pd.read_csv("fitness_health_tracking.csv")
        
        features = ['Age', 'Exercise_Hours_per_Week', 'Daily_Water_Intake_L', 
                    'Average_Sleep_Hours', 'Daily_Calories_Intake']
        target = 'Weight_Loss_1_Month_kg'
        
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns in dataset: {missing_cols}")
            return None
        
        reg = LinearRegression()
        X = df[features]
        y = df[target]
        reg.fit(X, y)
        
        user_data = pd.DataFrame([[age, exercise, water, sleep, calories]], columns=features)
        prediction = reg.predict(user_data)
        
        return prediction[0]
    
    except FileNotFoundError:
        print("Error: 'fitness_health_tracking.csv' file not found.")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_user_input():
    try:
        print("=== Weight Loss Prediction System ===")
        print("Please enter your information:")
        
        age = int(input("Enter your age: "))
        exercise = float(input("Enter your exercise hours per week: "))
        water = float(input("Enter your daily water intake in liters: "))
        sleep = float(input("Enter your average sleep hours per day: "))
        calories = int(input("Enter your daily calorie intake: "))
        
        return age, exercise, water, sleep, calories
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return None

def give_suggestions(prediction):
    print("\n=== Suggestions ===")
    if prediction < 0:
        print("⚠ You might gain weight. Consider:")
        print("- Reducing calorie intake slightly.")
        print("- Adding a 20-30 min morning walk.")
        print("- Drinking more water and improving sleep.")
    elif prediction < 0.5:
        print("✅ You're stable, but minor changes could improve weight loss.")
    else:
        print("🎉 Keep up the good work!")

def log_user_data(age, exercise, water, sleep, calories, prediction):
    filename = "user_progress_log.csv"
    headers = ['Date', 'Age', 'Exercise', 'Water', 'Sleep', 'Calories', 'Prediction']
    row = [datetime.now().strftime("%Y-%m-%d"), age, exercise, water, sleep, calories, prediction]
    
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(headers)
            writer.writerow(row)
    except Exception as e:
        print(f"Error logging data: {e}")

def compare_with_past():
    try:
        df = pd.read_csv("user_progress_log.csv")
        last_entries = df.tail(7)
        if not last_entries.empty:
            avg_pred = last_entries['Prediction'].mean()
            print(f"\n📊 Average weight loss (last 7 days): {avg_pred:.2f} kg")
        else:
            print("\n📊 Not enough data to compare trends.")
    except FileNotFoundError:
        print("\n📊 No historical data found. Start logging today!")

def journal_entry():
    entry = input("\n📝 Write a short journal for today (or press Enter to skip): ")
    if entry:
        with open("user_journal.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d')} - {entry}\n")
        print("Journal saved.")

def plot_trend():
    try:
        df = pd.read_csv("user_progress_log.csv")
        if df.shape[0] < 2:
            print("\n📈 Not enough data for trend visualization.")
            return
        print("\n=== Weight Loss Trend ===")
        last_preds = df['Prediction'].tail(10).tolist()
        for i, val in enumerate(last_preds, 1):
            bar = '#' * int(max(0, val * 2))  # scale factor for visibility
            print(f"Day {i}: {bar} ({val:.2f} kg)")
    except:
        print("\n📈 Unable to display trend chart.")

def main():
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
        give_suggestions(prediction)
        log_user_data(age, exercise, water, sleep, calories, prediction)
        compare_with_past()
        plot_trend()
        journal_entry()
        
        print("\n📌 Note: This is a prediction based on historical data and may not reflect individual variations.")
        print("\n📌 Note: All the data are generated by an AI.")
    else:
        print("❌ Failed to make prediction. Please check your data file and try again.")

if __name__ == "__main__":
    main()
