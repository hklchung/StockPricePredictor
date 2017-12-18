# Stock-Price-Predictor
# The project's domain background
The finance sector is a fascinating field to explore the power of machine learning and its application is well researched and documented. In particular, this field is often associated with high monetary compensations which is attractive to many of us to try our hands on applying the knowledge of mathematical and statistical models, and to mine patterns in stock movements so that we can make predictions and have a competitive edge over other traders.
Today, there are many Algo-trade brokers taking advantage of high-frequency financial data and electronic trading tools that are built on the foundations of machine learning. These high-frequency traders are often characterised by high speed, high turn-over rates and high order-to-trade ratios and are slowly replacing the traditional traders (Aldridge I., 2013). Algo-trading are often heavily leaned towards technical analysis as opposed to fundamental analysis, this means that indicators of buy and sell opportunities are often built on only information of historical price and volume, rather than traditional valuation of a company (Moldovan D. et al, 2011).
Since my current knowledge does not allow myself to carry out meaningful analysis of a company’s value and subsequently perform valuation of its stock, it was found that this particular method of stock price prediction to be a highly suitable area to study.
The data was collected from “Yahoo! Finance”. Datasets of historical data of stocks from “Yahoo! Finance” typically has the following structure: Date, Open, High, Low, Close, AdjClose and Volume. However, upon inspection the volume feature was found to be loosely recorded which made this feature particularly unreliable and impractical for analysis.
Since today’s global markets are all interconnected and the fall of one market will most likely signal the other to follow, I have also gather the datasets of other large financial market indexes including the ASX 200 (Sydney, Australia), CAC40 (Paris, France), DAX (Frankfurt, Germany), DJI (New York, USA), HSI (Hong Kong), Nikkei225 (Tokyo, Japan) and NASDAQ (New York, USA). I have sourced data from 1992 to present which would provide this project a large enough timeframe in order for meaningful analysis to take place.

# A problem statement
The proposed problem is to use machine learning to analyse time-series financial data to successfully build a model that can predict the price movement of S&P500 stock index better than a naïve predictor. In this case, a naïve predictor is one that predicts the stock index using a rolling average with a 10 day window. Specifically, this presents as a supervised classification problem where the output would be either 0 for decrease/no change of daily return, or 1 for increase of daily return. The performance of the trained model can then be evaluated using accuracy and F1-score, which would give us a more robust evaluation of the trained model’s performance.

# Methodology
The workflow of the project goes as follows:
1) Extract and Clean the data from Yahoo Finance
2) Feature Engineering with Dimensionality Reduction
3) Train with Classification Algorithms
4) Train with Optimized Classification Algorithms with GridSearch using TimeSeriesSplit for cross validation

To see how each stage of how the model was built, please check the Jupyter Notebooks:
1) Step1_Data+Extraction+and+Cleaning.ipynb
2) Step2_Feature+Engineering+and+Dimensionality+Reduction.ipynb
3) Step3_Machine+Learning+Implementation+and+Optimization.ipynb

To see more in depth comments and coding, please visit the Python scripts folder.

# Future Directions
I would like to point out that neural networks were not considered for this project due to the computational demands of these algorithms. In particular, I felt that recurrent neural networks such as Long-Short Term Memory (LSTM) neural networks were the most appropriate for this project as these models were designed specifically for time-series data. Also, due to time constraints other methods of feature engineering were not explored, classic stock market indicators such as Bollinger Bands, Sharpe Ratios and even more days in the rolling period alone may benefit the learners significantly. This concludes the first stage of this project but the above mentioned room for improvements will be explored in the second stage of this project.
