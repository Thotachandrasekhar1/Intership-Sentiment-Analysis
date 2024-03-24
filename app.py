from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


###########################################

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    review = request.form.get("Review")

    model = joblib.load("model/naive_bayes.pkl")

    prediction = model.predict([review])[0]

    sentiment_label = 'Positive' if prediction == 1 else 'Negative'

    return render_template("output.html", sentiment_label=sentiment_label)



###########################################

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)