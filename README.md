import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class UniversityAdmissionPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic sample data for university admissions"""
        np.random.seed(42)
        
        # Generate features
        gpa = np.random.normal(3.2, 0.5, n_samples)
        gpa = np.clip(gpa, 0.0, 4.0)  # Clip to valid GPA range
        
        sat_score = np.random.normal(1200, 200, n_samples)
        sat_score = np.clip(sat_score, 400, 1600)  # Valid SAT range
        
        extracurriculars = np.random.randint(0, 10, n_samples)
        volunteer_hours = np.random.exponential(50, n_samples)
        essays_score = np.random.uniform(1, 10, n_samples)
        
        # Create admission probability based on weighted features
        admission_score = (
            0.4 * (gpa / 4.0) +
            0.3 * (sat_score / 1600) +
            0.1 * (extracurriculars / 10) +
            0.1 * (np.minimum(volunteer_hours, 200) / 200) +
            0.1 * (essays_score / 10)
        )
        
        # Add some randomness and create binary admission decision
        admission_score += np.random.normal(0, 0.1, n_samples)
        admitted = (admission_score > 0.6).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'GPA': gpa,
            'SAT_Score': sat_score,
            'Extracurriculars': extracurriculars,
            'Volunteer_Hours': volunteer_hours,
            'Essay_Score': essays_score,
            'Admitted': admitted
        })
        
        return data
    
    def train_model(self, data=None):
        """Train the admission prediction model"""
        if data is None:
            data = self.generate_sample_data()
        
        # Prepare features and target
        features = ['GPA', 'SAT_Score', 'Extracurriculars', 'Volunteer_Hours', 'Essay_Score']
        X = data[features]
        y = data['Admitted']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return data
    
    def predict_admission_probability(self, gpa, sat_score, extracurriculars, 
                                    volunteer_hours, essay_score):
        """Predict admission probability for a single student"""
        if not self.is_trained:
            print("Model not trained yet. Training with sample data...")
            self.train_model()
        
        # Prepare input data
        student_data = np.array([[gpa, sat_score, extracurriculars, 
                                volunteer_hours, essay_score]])
        student_data_scaled = self.scaler.transform(student_data)
        
        # Get probability
        probability = self.model.predict_proba(student_data_scaled)[0][1]
        prediction = self.model.predict(student_data_scaled)[0]
        
        return probability, prediction
    
    def analyze_student_profile(self, gpa, sat_score, extracurriculars, 
                              volunteer_hours, essay_score):
        """Provide detailed analysis of student's admission chances"""
        probability, prediction = self.predict_admission_probability(
            gpa, sat_score, extracurriculars, volunteer_hours, essay_score
        )
        
        print("=" * 50)
        print("UNIVERSITY ADMISSION ANALYSIS")
        print("=" * 50)
        print(f"Student Profile:")
        print(f"  • GPA: {gpa:.2f}/4.0")
        print(f"  • SAT Score: {sat_score}")
        print(f"  • Extracurricular Activities: {extracurriculars}")
        print(f"  • Volunteer Hours: {volunteer_hours}")
        print(f"  • Essay Score: {essay_score:.1f}/10")
        print()
        print(f"Admission Probability: {probability:.1%}")
        print(f"Predicted Decision: {'ADMITTED' if prediction else 'NOT ADMITTED'}")
        print()
        
        # Provide recommendations
        print("RECOMMENDATIONS:")
        if gpa < 3.5:
            print("  • Focus on improving GPA - consider retaking courses")
        if sat_score < 1300:
            print("  • Consider retaking SAT/ACT for better scores")
        if extracurriculars < 3:
            print("  • Increase involvement in extracurricular activities")
        if volunteer_hours < 50:
            print("  • Engage in more community service")
        if essay_score < 7:
            print("  • Spend more time crafting compelling personal essays")
        
        if probability > 0.7:
            print("  • Strong profile! Apply to reach schools as well")
        elif probability < 0.3:
            print("  • Consider safety schools and gap year options")
        
        return probability, prediction

def main():
    """Main function to demonstrate the admission predictor"""
    predictor = UniversityAdmissionPredictor()
    
    print("Training University Admission Predictor...")
    print("-" * 40)
    
    # Train the model
    data = predictor.train_model()
    
    print("\n" + "=" * 60)
    print("SAMPLE STUDENT PREDICTIONS")
    print("=" * 60)
    
    # Test with sample students
    sample_students = [
        {"name": "Alice", "gpa": 3.8, "sat": 1450, "extra": 5, "volunteer": 120, "essay": 8.5},
        {"name": "Bob", "gpa": 3.2, "sat": 1200, "extra": 2, "volunteer": 30, "essay": 6.0},
        {"name": "Carol", "gpa": 3.9, "sat": 1520, "extra": 7, "volunteer": 200, "essay": 9.2},
        {"name": "David", "gpa": 2.8, "sat": 1050, "extra": 1, "volunteer": 10, "essay": 5.5}
    ]
    
    for student in sample_students:
        print(f"\nAnalyzing {student['name']}:")
        predictor.analyze_student_profile(
            student['gpa'], student['sat'], student['extra'], 
            student['volunteer'], student['essay']
        )
        print("-" * 50)
    
    # Interactive prediction
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION")
    print("=" * 60)
    print("Enter your details for admission probability:")
    
    try:
        gpa = float(input("Enter GPA (0.0-4.0): "))
        sat_score = int(input("Enter SAT Score (400-1600): "))
        extracurriculars = int(input("Enter number of extracurricular activities: "))
        volunteer_hours = float(input("Enter volunteer hours: "))
        essay_score = float(input("Enter essay score (1-10): "))
        
        predictor.analyze_student_profile(
            gpa, sat_score, extracurriculars, volunteer_hours, essay_score
        )
        
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default example instead...")
        predictor.analyze_student_profile(3.5, 1300, 3, 75, 7.0)

if __name__ == "__main__":
    main()
