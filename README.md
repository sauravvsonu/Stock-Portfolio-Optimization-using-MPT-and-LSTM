CS229 INNOVATION LAB REPORT, IIT PATNA

**Stock Portfolio Optimization using LSTM and MPT**

By- Shivanshu Sanjeev(1901CS56) and Saurav Sonu(1901MM30)

Under the supervision of Prof. Somnath Tripathy, IIT Patna

**Introduction  

**

Stock markets have existed for centuries. Since its inception, stock markets
have served many purposes, the most important being to provide companies with a
source to raise capital for investment and expansion. For the individual
investor, the stock market provides a way to invest your income to earn a share
of the companies’ profits. And for a risk averse investor the most important
thing is to manage a portfolio.

Portfolio management refers to managing an individual’s investments in the form
of bonds, shares, cash, mutual funds etc so that he earns the maximum profits
within the stipulated time frame. Portfolio management **minimizes the risks**
involved in investing and also increases the chance of making profits.

So, the goal of this project is to build an **optimized portfolio for
investment** in the stock market where the investor will provide the names of
certain number stocks and the model will return the best possible combination of
shares amongst the top-performing stocks which will be first predicted by the
**LSTM model** and then the selected stocks will be optimized using **Markowitz
Portfolio Theory.**

**Existing Work And Our Innovation**

There are numerous classical methods to build a portfolio like Dow Jones theory,
Random Walk theory, Formula theory and then the most accurate one is **Markowitz
Portfolio theory.**

But since Markowitz portfolio theory considers covariance and correlation
between two stocks as a measure of risk, it overlooks the possibility of
unwanted and unexpected changes in stock prices such as downside risks.

So to back up the decision of what stocks we should invest in, we first predict
the behaviour of stocks with the help of **LSTM**, by doing this we have
decreased the risk of choosing the unwanted stocks even further as MPT alone,
only decides on the basis previous performance but LSTM will feed in the the
predicted future data to the MPT model which makes it more accurate and less
risky for investors.

-   **How LSTM is Helping make a better stock choice for a higher return in
    lesser time  
    **As accurate as MPT is for portfolio optimization, its dependency on
    covariance and correlation sometimes tend to overlook the sudden changes in
    the market price also called downside risk.   
    **Two portfolios that have the same level of variance and returns are
    considered equally desirable under modern portfolio theory. One portfolio
    may have that variance because of frequent small losses. In contrast, the
    other could have that variance because of rare spectacular declines.** Even
    though the two stocks behaviour might show satisfactory correlation
    according to MPT, it might result in overall loss in a shorter span of time.

    **What LSTM will do is** predict such a sudden change and will discard the
    risky stock well before passing it into the MPT model and not only it will
    satisfy the diversity of MPT but also takes care of sudden decline in stock
    value making the model more accurate and useful for investment in a shorter
    span of time and on a higher frequency.

**THEORY**

-   **LSTM**

The Long short-term memory (LSTM) is a special type of Recurrent Neural Networks
(RNN) architecture that is mainly used in deep learning. LSTM uses a set of
feedback connections to process sequences of data. This architecture is known
for its efficiency in making predictions, processing, and classifying
large-scale time-series data despite the presence of some lags between events.
It was named long short-term memory because its cell unit has the ability to
forget a part of previously stored data and can, at the same time, memorize
additional new pieces of information.

An LSTM unit encapsulates the following elements:

• Cell: represents the memory part of the LSTM that monitors the dependencies
between different elements constituting the input sequence.

• Input gate: regulates the information flowing into the cell.

• Output gate: regulates the information flowing out of the cell.

• Forget gate: remember the different values over a time interval.

![](media/8589d58633a7113ac61c1134da6923da.png)

![](media/f7ff4bc699f8747a21f823ca6617bf62.png)

-   **Adam Optimizer**

The type of optimizer used can greatly affect how fast the algorithm converges
to the minimum value. Here we have chosen to use Adam optimizer. The Adam
optimizer combines the perks of two other optimizers: ADAgrad and RMSprop.

The stochastic gradient descent update for RMS prop is given by:

![](media/51c1dcb689b71d30136482125100af4a.png)

Adam(Adaptive Movement Estimation) is another method that computes the adaptive
learning rates for each parameter based on its past gradients.

![](media/6ad2862da4e3e6ae2cd8c0257fc8ffa2.png)![](media/d70de68db032333fc8604313aed4cd9a.png)

The minimization problem is given by:

![](media/507bb571033d7b2bbdb361d086612490.png)

Another important aspect of training the model is making sure the weights do not
get too large, hence, overfit. For this purpose, we chose to do regularization.

-   **MPT**

Modern [portfolio](https://www.investopedia.com/terms/p/portfolio.asp) theory
(MPT) is a theory on how risk-averse investors can construct portfolios to
maximize [expected
return](https://www.investopedia.com/terms/e/expectedreturn.asp) based on a
given level of market risk. [Harry
Markowitz](https://www.investopedia.com/terms/h/harrymarkowitz.asp) pioneered
this theory in his paper “Portfolio Selection”.

Modern portfolio theory argues that an investment's risk and return
characteristics should not be viewed alone, but should be evaluated by how the
investment affects the overall portfolio's risk and return. MPT shows that an
investor can construct a portfolio of multiple assets that will maximize returns
for a given level of risk.

Likewise, given a desired level of expected return, an investor can construct a
portfolio with the lowest possible risk. Based on statistical measures such as
[variance](https://www.investopedia.com/terms/v/variance.asp) and
[correlation](https://www.investopedia.com/terms/c/correlation.asp), an
individual investment's performance is less important than how it impacts the
entire portfolio. MPT assumes that investors are risk-averse, meaning they
prefer a less risky portfolio to a riskier one for a given level of return.

-   **Efficient Frontier Curve and Max Sharpe Ratio**

The Efficient Frontier is a common phrase in Modern Finance since the inception
of Modern Portfolio Theory in 1952 by Harry Markowitz.

The efficient frontier, a cornerstone of modern portfolio theory, shows the set
of portfolios that provide the highest level of return for the lowest level of
risk. When a portfolio falls to the right of the efficient frontier, they
possess greater risk relative to their return. Conversely, when a portfolio
falls beneath the slope of the efficient frontier, they offer a lower level of
return relative to risk.

![](media/f582cc985d2ccc9a8851ee873adbbc6a.png)

Owning a portfolio with the Maximum Sharpe Ratio is the dream of every investor
("or more likely should be")!!!

Sharpe Ratio - It is the average return earned in excess of the risk-free rate
per unit of [volatility](https://www.investopedia.com/terms/v/volatility.asp) or
total risk. Volatility is a measure of the price fluctuations of an asset or
portfolio.

**So what is the dream portfolio and how can you make one?**

According to Markowitz's theory, there is an optimal portfolio that could be
designed with a perfect balance between risk and return.

![](media/cbf1acbb0ab6b273a2c5edab7c59346d.png)![](media/cbf1acbb0ab6b273a2c5edab7c59346d.png)

**IMPLEMENTATION**

1.  First our Goal is to fetch the current stock data using Yahoo Finance API.
    The fetched data will keep on changing as the market changes because it
    always gives present day stock values and hence it was chosen instead of
    reading data from CSV files.

![](media/be0dd3524f29a339c93f0f94ab1789db.png)
![](media/0229b48f4162568ae83ec953eeaab75b.png)

**![](media/2225b7956c419b47578615fafcf33cf6.png)**

1.  The Imported data is Now Plotted, processed and regularised for using it
    efficiently in a single LSTM trainer. Also, Train Data and Test data is
    stored separately.

![](media/4714a8a9e818478179c8cec696802314.png)![](media/4e842c0844251c83ab4c454ebd84d8a1.png)![](media/870be1735e5cb780ad9d230967e84d64.png)

1.  To improve the speed of the model we will first optimise it with the help of
    Adam optimiser. Without it just to run all the epochs for a single company
    stock it takes more time than all the stock predicted at once using Adam
    Optimiser.

![](media/ef6d190460fde19366efe142601f2fae.png)

1.  Now the Data is ready to train and predict the values which will be compared
    by test data later on. Graphs plotted and rms values are also calculated.

![](media/25db752760dcf6e7ee0010f3f1b255e8.png)

1.  After the prediction is made the predicted data give a certain return value
    for each stock share. We then took the Moving **Average** of daily return of
    stocks for next six months and then selected the top 5 best performing
    stocks.

![](media/ae8b77479f5bde458592925ac533bc09.png)

1.  Now that we have successfully selected the appropriate stocks to invest in
    we will proceed on building an Optimum Portfolio. First we will calculate
    the variance of each stock, and then **covariance** and leading to
    correlation between each pair of
    stocks.![](media/c7e1f92b0d4a8e822233af6b4289695f.png)

![](media/717861ab1ca770c8964c9e65b057d20c.png)

![](media/4388434963be9a79b0df70f4962057a6.png)

And then we plotted the correlation matrix of each stock.

1.  With 5000 randomized weights we now plotted the graph of return versus
    volatility and visualized the efficient frontier. And then we finally
    calculated the maximum return, minimum risk and maximum Sharpe ratio
    portfolios.

Hence, we finally receive our desired percentage of investment in each stock.

![](media/3773b79bb472ad7a35468479e295d590.png)Calculating Sharpe ratio

![](media/1bc69e21c40f9317912ee480995705ef.png)

**Results And Observation**

-   For LSTM the testing RMSE values we found were as low as 2.646, 3.524 and
    3.895 for ONGC, NCC and ITC stocks respectively which is pretty accurate to
    predict future values of stock. Given aside are the closing actual and
    predicted values of Wipro Stock given by the LSTM model.

![](media/7c877b76a80a8c9ca8dab7d66e055943.png)![](media/a9ecef16419d147e028069d07bfd54d6.png)

![](media/3d9442b4aede1b8be824381f0a7e6eda.png)

-   For MPT below is the correlation matrix and efficient frontier graph
    observed.

    By the table below, we can get the max return of 16.085% with a volatility
    of 25.235% by investing the highest 65.127% in HDFC.It also shows that we
    can get the return of 4.919% at the min risk possible by investing the
    highest 26.631% in Coal India.

![](media/2921a732eb5399a5ab7bb8e7cb4f0c8a.png)![](media/d9c2e32b2c24ed0721d286caaa696a9f.png)![](media/22dceb5d76b562f5d1c20618010542c6.png)

![](media/4e81129f298958c2761dac143fe70e3f.png)

**Hence, we successfully built a risk averse portfolio and maximum sharpe ratio
portfolio. The portfolio we get is Dynamic and can be used for high frequency
tradings.**

**CONCLUSION**

Through this project, we learnt how to apply deep learning techniques such as
LSTM and Linear Regression Models, how to use keras-tensorflow library, how to
collect and pre-process given data, how to analyse model's performance and
optimise LSTM Network and to ensure increase in positive results. We built a
model to accurately predict the future closing price of a given stock, using the
Long Short-Term Memory Neural Net algorithm.

And also using the predicted stock prices from the LSTM, we were able to make an
optimized portfolio which can help investors in making risk-averse decisions
using Markowitz Portfolio theory.

**References**

[**Long Short Term Memory (LSTM) - Recurrent Neural Networks \|
Coursera**](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)

[**Modern Portfolio Theory Explained! -
YouTube**](https://www.youtube.com/watch?v=8jKrnUfIYEg)

[**Modern Portfolio Theory (MPT)
(investopedia.com)**](https://www.investopedia.com/terms/m/modernportfoliotheory.asp#:~:text=Modern%20portfolio%20theory%2C%20introduced%20by%20Harry%20Markowitz%20in,risk%20level%20for%20the%20same%20level%20of%20return.)

[**LSTM Networks \| A Detailed Explanation \| Towards Data
Science**](https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9)
