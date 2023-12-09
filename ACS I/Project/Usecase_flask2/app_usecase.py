from flask import Flask, request, render_template,jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle 
import numpy as np
import json
import pandas as pd
# import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# python anywhere
# currently using logistic regression for both coz of disk issue
# make jupyter notebook ready 

# web: gunicorn app:app (first app app_usecase.py second app is flask instance)
app = Flask(__name__)

# pickle files load

#race
with open(r"./artifacts/race_encoded.pkl", 'rb') as file:
    race_model_data = pickle.load(file)

#gender
with open(r"./artifacts/gender_encoded.pkl", 'rb') as file:
    gender_model_data = pickle.load(file)

#age
with open(r"./artifacts/age_scaled.pkl", 'rb') as file:
    age_data = pickle.load(file)

#weight
with open(r"./artifacts/weight_scaled.pkl", 'rb') as file:
    weight_data = pickle.load(file)

# A1C_result
with open(r"./artifacts/A1Cresult_encoded.pkl", 'rb') as file:
    A1C_result_model_data = pickle.load(file)

#max_glu_serum
with open(r"./artifacts/max_glu_serum_encoded.pkl", 'rb') as file:
    max_glu_serum_model_data = pickle.load(file)

# tf_id_vectorizer symptoms
with open(r"./artifacts/tfidf_vectorizer.pkl", 'rb') as file:
    symptoms_tfid = pickle.load(file)

# xg boost 
with open(r"./artifacts/xgb_model.pkl", 'rb') as file:
    xgb_data = pickle.load(file)

# xgb column names 
with open(r"./artifacts/column_names_xgb.pkl", 'rb') as file:
    xgb_columns= pickle.load(file)

# diabetes_random_forest_Classifier
with open(r"./artifacts/type_ran_classi_model.pkl", 'rb') as file:
    type_ran_classi_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/preprocess', methods=['POST'])
def preprocess():
    drug_mapping = {0: 'Metformin',1: 'Glipizide',2: 'Glyburide',3: 'Insulin'}

    diabetes_type_mapping = {0: 'Type_1',1: 'Type_2'}

    description={'Metformin':' The active ingredient metformin hydrochloride, each tablet contains the following inactive ingredients: candellila wax, cellulose acetate, hypromellose, magnesium stearate, polyethylene glycols (PEG 400, PEG 8000), polysorbate 80, povidone, sodium lauryl sulfate, synthetic black iron oxides, titanium dioxide, and triacetin.',
                 'Glipizide':' Each tablet contains the following inactive ingredients: anhydrous lactose, microcrystalline cellulose, corn starch, silicon dioxide, stearic acid.',
                 'Glyburide':' Glyburide tablets contain the active ingredient glyburide and each tablet contains the following inactive ingredients: dibasic calcium phosphate USP, magnesium stearate NF, microcrystalline cellulose NF, sodium alginate NF, talc USP.',
                 'Insulin':' he active ingredient insulin lispro protamine suspension and insulin lispro. Inactive ingredients: protamine sulfate, glycerin, dibasic sodium phosphate, metacresol, zinc oxide (zinc ion), phenol and water for injection.'}
    
    
    
    data = request.json

    preprocessing_result = eda(data)
    prediction,type_dia=predict(preprocessing_result)

    drug_recommendation = drug_mapping.get(prediction, 'Unknown Drug')
    diabetes_type = diabetes_type_mapping.get(type_dia, 'Unknown Type')

    return jsonify({"Drug_recommended": drug_recommendation,
                    "Diabetes_type":diabetes_type,
                    "Description":description[str(drug_recommendation)]})

@app.route('/logresults', methods=['POST'])
def log():
    data = request.data.decode('utf-8')
    print("data",data)
    file = open("logs.csv", "a") # append mode
    file.write(data+"\n")
    file.close()
    return jsonify({"success": True})

# search
# 
def eda(data):

    overall_data_array1=[]
    overall_data_array2=[]
    data = request.json 

    race_encoder = race_model_data['race_encoder'] 
    race_value = data.get('race')
    encoded_race = race_encoder.transform([race_value])[0]
    overall_data_array1.append(encoded_race)


    gender_encoder = gender_model_data['gender_encoder']
    gender_value = data.get('gender')
    encoded_gender = gender_encoder.transform([gender_value])[0]
    overall_data_array1.append(encoded_gender)
    # print(encoded_gender)
    # print(type(encoded_gender))

    age_encoder = age_data['age_scaler']
    age_value = data.get('age')
    encoded_age = age_encoder.transform([[age_value]])[0]
    overall_data_array1.append(encoded_age[0])
    # print(encoded_gender)
    # print(type(encoded_gender))

    weight_encoder = weight_data['weight_scaler']
    weight_value = data.get('weight')
    encoded_weight = weight_encoder.transform([[weight_value]])[0]
    overall_data_array1.append(encoded_weight[0])



    A1C_result_encoder = A1C_result_model_data['A1Cresult_encoder']
    A1C_result_value = data.get('A1C_result')
    encoded_A1C_result = A1C_result_encoder.transform([A1C_result_value])[0]
    overall_data_array1.append(encoded_A1C_result)



    max_glu_serum_encoder = max_glu_serum_model_data['max_glu_serum_encoder']
    max_glu_serum_value = data.get('max_glu_serum')
    encoded_max_glu_serum = max_glu_serum_encoder.transform([max_glu_serum_value])[0]
    overall_data_array1.append(encoded_max_glu_serum)


    symptoms_tfid_encoder = symptoms_tfid['tfidf_vectorizer']
    symptoms_tfid_value = data.get('symptoms')
    encoded_symptoms_tfid_encoder = symptoms_tfid_encoder.transform([symptoms_tfid_value])


    encoded_symptoms_tfid_encoder_array=encoded_symptoms_tfid_encoder.toarray()[0]
    serializable_list = encoded_symptoms_tfid_encoder_array.tolist()

    json_data = json.dumps(serializable_list)
    print(type(json_data))
    overall_data_array2.append(json_data)

    #it=pd.concat([pd.Series(overall_data_array1),pd.Series(json_data)],axis=0)

    
    overall_data_array2=pd.DataFrame(overall_data_array2)
    int_values=pd.DataFrame(overall_data_array1)
    #overall_data_array2.to_csv(r"C:/Users/LENOVO/Desktop/Files/NutaNXT/Work/Training/All_flask/Usecase_flask/dtaackjsakdc.csv")
    #tfidf_values=json.loads(overall_data_array2['0'][0])
    tfidf_values=json.loads(json_data)
    result=pd.concat([int_values,pd.Series(tfidf_values)],axis=0)
    # result.to_csv(r"C:/Users/LENOVO/Desktop/Files/NutaNXT/Work/Training/All_flask/Usecase_flask/dtaackjsakdc.csv")
    #result=result.T
    
    # return jsonify({'encoded_race': int(encoded_race),
    #                 'encoded_gender': int(encoded_gender),
    #                 'encoded_age':float(np.round(encoded_age,4)),
    #                 'encoded_weight':float(np.round(encoded_weight,4)),
    #                 'encoded_A1C_result': int(encoded_A1C_result),
    #                 'encoded_max_glu_serum': int(encoded_max_glu_serum),
    #                 'encoded_symptoms_tfid_encoder':tfidf_values
    #                 })
    
    return result.values.ravel().tolist()

def predict(result):
    xgb_classifier = xgb_data['xgb_model']
    result = pd.DataFrame(result)
    result = result.T
    result.columns = xgb_columns
    prediction = int(xgb_classifier.predict(result)[0])

    type_classifier=type_ran_classi_model['diabetes_type_model']
    type_dia=int(type_classifier.predict(result)[0])
    return (prediction,type_dia)




if __name__ == '__main__':
    app.run(debug=True)
