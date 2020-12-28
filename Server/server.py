from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__, static_url_path="/client", static_folder='../client', template_folder="../client")


@app.route('/', methods=['GET'])
def index():
    if request.method == "GET":
        return render_template("app.html")


@app.route('/getsex')
def get_gender():
    response = jsonify({
        'gender': util.get_gender()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    applicantincome = float(request.form['applicantincome'])
    coapplicantincome = float(request.form['coapplicantincome'])
    loanamount = float(request.form['loanamount'])
    loanterm = float(request.form['loanterm'])
    credithistory = float(request.form['credithistory'])
    gender = request.form['gender']
    status = request.form['status']
    dependent = request.form['dependent']
    education = request.form['education']
    employement = request.form['employement']
    propertyarea = request.form['propertyarea']

    response = jsonify({
        'estimated_price': util.get_estimated_price(applicantincome, coapplicantincome, loanamount, loanterm,
                                                    credithistory, gender, status, dependent,
                                                    education, employement, propertyarea)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run()
