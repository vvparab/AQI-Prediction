import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model0 = pickle.load(open('Linear_Regression.pkl', 'rb'))
model1 = pickle.load(open('Lasso(L1)_Ridge(L2)_Regression.pkl', 'rb'))
model2 = pickle.load(open('Regression_Decision_Tree.pkl', 'rb'))
model3 = pickle.load(open('regression_random_forest.pkl', 'rb'))
model4 = pickle.load(open('Regression_xgboost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction0 = model0.predict(final_features)
    
    output0 = np.round(prediction0[0], 2)

    prediction1 = model1.predict(final_features)
    output1 = np.round(prediction1[0], 2)

    prediction2 = model2.predict(final_features)
    output2 = np.round(prediction2[0], 2)

    prediction3 = model3.predict(final_features)
    output3 = np.round(prediction3[0], 2)

    prediction4 = model4.predict(final_features)
    output4 = np.round(prediction4[0],2)

    return render_template('index.html', 
                           prediction_text0='Predicted Air Quality Index (Linear) : {}'.format(output0),
                           prediction_text1='Predicted Air Quality Index (L1 L2) : [{}]'.format(output1),
                           prediction_text2='Predicted Air Quality Index (D_Tree) : [{}]'.format(output2),
                           prediction_text3='Predicted Air Quality Index (R_Forest) : [{}]'.format(output3),
                           prediction_text4='Predicted Air Quality Index (XgBoost) : [{}]'.format(output4))    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction0 = model0.predict([np.array(list(data.values()))])
    prediction1 = model1.predict([np.array(list(data.values()))])    
    prediction2 = model2.predict([np.array(list(data.values()))])
    prediction3 = model3.predict([np.array(list(data.values()))])
    prediction4 = model4.predict([np.array(list(data.values()))])

    output0 = prediction0[0]
    return jsonify(output0)

    output1 = prediction1[0]
    return jsonify(output1)
    
    output2 = prediction2[0]
    return jsonify(output2)

    output3 = prediction3[0]
    return jsonify(output3)
    
    output4 = prediction4[0]
    return jsonify(output4)
    

if __name__ == "__main__":
    app.run(debug=True)