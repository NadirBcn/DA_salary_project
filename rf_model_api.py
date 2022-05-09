
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('rf_model_api')

# Define predict function
@app.post('/predict')
def predict(index, Job_Title, Salary_Estimate, Job_Description, Rating, Company_Name, Location, Size, Founded, Type_of_ownership, Industry, Sector, Revenue, Hourly, Employer_provided, Min_Salary, Max_Salary, Company_text, State, Age, Python, Tableau, Excel, Power_BI, SAS, SQL, SSIS, Job_Simplified, Seniority, Desc_length):
    data = pd.DataFrame([[index, Job_Title, Salary_Estimate, Job_Description, Rating, Company_Name, Location, Size, Founded, Type_of_ownership, Industry, Sector, Revenue, Hourly, Employer_provided, Min_Salary, Max_Salary, Company_text, State, Age, Python, Tableau, Excel, Power_BI, SAS, SQL, SSIS, Job_Simplified, Seniority, Desc_length]])
    data.columns = ['index', 'Job Title', 'Salary Estimate', 'Job Description', 'Rating', 'Company Name', 'Location', 'Size', 'Founded', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Hourly', 'Employer provided', 'Min Salary', 'Max Salary', 'Company_text', 'State', 'Age', 'Python', 'Tableau', 'Excel', 'Power BI', 'SAS', 'SQL', 'SSIS', 'Job Simplified', 'Seniority', 'Desc_length']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)