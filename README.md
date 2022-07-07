
# Startup Profit Prediction

* In this project, we build a simple ML model (multiple-linear regression) for the Kaggle dataset 50_startups.csv.
* The model can predict the profit a startup can generate, given it's investments on R&D, administration, and marketing along with the state it is located at.
* Our model can make accurate predictions upto an R-squared value of 98%. 
* We are in a mission to make investors, and founders smarter; give it a try....

## Acknowledgements
This project is based on the 50 startups public dataset available on Kaggle.

 - [Dataset](https://www.kaggle.com/datasets/farhanmd29/50-startups)
 


## Authors
- [@kaveenjayamanna](https://github.com/ktjayamanna)


## Installation

Install using pip

```bash
$ pip install -r requirements
```

## Deployment
To access the live demo, click here (https://startup50predictorv4.herokuapp.com/)

To deploy this project run

```bash
$ python fiftyStartUps.py
```



## Analysis

After identifying the dependent and independent variables, we check the correlation between dependent variables and ind. variables. 
Notice that both R&D, and marketing shows a strong positive correlation with profits. It makes sense right? You need a great product that is perfected over time as well as a killer marketing strategy to
get it out there to the users/ customers.

![Correlation check](https://github.com/ktjayamanna/startupProfitPrediction/blob/main/plots/correlation.png)

It is also important that we check for multicollinearity. Hmm..., there's some correlation between R&D and marketing. Maybe the more R&D a startup invests in, the more stuff to show off through marketing channels?

![multicollinearity check](https://github.com/ktjayamanna/startupProfitPrediction/blob/main/plots/multicollinearity.png)

After building the model, we tested with some data and plotted the predictions along with the test labels. Notice that we did that for all three numerical features. The big green line is the mathematical equation for our model.

* R&D against profits

![R&D](https://github.com/ktjayamanna/startupProfitPrediction/blob/main/plots/R%26D%20Spend.png)

* Administration against profits

![Admin](https://github.com/ktjayamanna/startupProfitPrediction/blob/main/plots/Administration.png)

* marketing against profits

![Marketing](https://github.com/ktjayamanna/startupProfitPrediction/blob/main/plots/Marketing%20Spend.png)




