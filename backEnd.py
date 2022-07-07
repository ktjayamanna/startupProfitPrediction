import pandas as pd
import numpy as np
import pickle

# from flask import Flask, request, url_for, redirect, render_template, jsonify
from flask import Flask, request, render_template
import os

app = Flask(__name__)
pickle_input = open("startup_50.pkl", "rb")
model = pickle.load(pickle_input)


def normalize_feature(featureIn, featureIdxIn):
    feature = float(featureIn)
    maxVals = [165349.2, 182645.56, 471784.1]
    if feature > maxVals[featureIdxIn]:
        return 1
    else:
        return feature / maxVals[featureIdxIn]


# homepage
@app.route("/")
def homePage():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    if int_features[-1] == "Other":
        return render_template(
            "home.html",
            pred="Sorry!, we do not cover your area yet, please check again in the future!",
        )
    else:
        print(int_features)
        if int_features[-1] == "New York":
            int_features[-1] = 0
            int_features.append(1)
        elif int_features[-1] == "Florida":
            int_features[-1] = 1
            int_features.append(0)
        else:
            int_features[-1] = 0
            int_features.append(0)
        for i in range(3):
            int_features[i] = normalize_feature(int_features[i], i)
        prediction = model.predict([int_features])
        prediction = prediction[0]
        return render_template(
            "home.html",
            pred="The profit of your company is ${}".format(round(prediction[0], 2)),
        )


# run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8011)
