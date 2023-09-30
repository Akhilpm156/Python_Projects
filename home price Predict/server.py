from flask import Flask,request, jsonify
import util

app = Flask(__name__)

@app.route('/get_home_price',methods = ['POST'])

def get_home_price():
    
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin','*')

    return response

if __name__ == "__main__":
    print("starting Python Flask Server for Home price prediction....")
    util.load_saved_artifacts()
    app.run()