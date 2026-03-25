from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


model = joblib.load('GradientBoosting_VCB.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            
            open_p = float(request.form['open'])
            high_p = float(request.form['high'])
            low_p  = float(request.form['low'])
            vol    = float(request.form['volume'])
            ret1   = float(request.form['return1d'])
            ma5    = float(request.form['ma5'])
            ma20   = float(request.form['ma20'])
            
            
            X_new = pd.DataFrame([{
                'OPEN': open_p, 'HIGH': high_p, 'LOW': low_p,
                'VOLUME': vol, 'Return_1d': ret1, 'MA_5': ma5, 'MA_20': ma20
            }])
            
            # Dự báo
            pred = model.predict(X_new)[0]
            prediction = round(pred, 2)
            
        except Exception as e:
            prediction = f"Lỗi: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
