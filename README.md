# Stock Portfolio Optimization using LSTM and MPT

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

## Existing Work And Our Innovation

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

## CONCLUSION

Through this project, we learnt how to apply deep learning techniques such as
LSTM and Linear Regression Models, how to use keras-tensorflow library, how to
collect and pre-process given data, how to analyse model's performance and
optimise LSTM Network and to ensure increase in positive results. We built a
model to accurately predict the future closing price of a given stock, using the
Long Short-Term Memory Neural Net algorithm.

And also using the predicted stock prices from the LSTM, we were able to make an
optimized portfolio which can help investors in making risk-averse decisions
using Markowitz Portfolio theory.

## References

[**Long Short Term Memory (LSTM) - Recurrent Neural Networks \|
Coursera**](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)

[**Modern Portfolio Theory Explained! -
YouTube**](https://www.youtube.com/watch?v=8jKrnUfIYEg)

[**Modern Portfolio Theory (MPT)
(investopedia.com)**](https://www.investopedia.com/terms/m/modernportfoliotheory.asp#:~:text=Modern%20portfolio%20theory%2C%20introduced%20by%20Harry%20Markowitz%20in,risk%20level%20for%20the%20same%20level%20of%20return.)

[**LSTM Networks \| A Detailed Explanation \| Towards Data
Science**](https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9)
