# Optimizers 
## Adam
on a given interation t, we can calculate moving averages based on parameters $\beta_{1}$ and $\beta_{2}$. $\beta_{1}$ is the exponential decay of the rate for the first moment estimates, and its literature value is 0.9. $\beta_{2}$ is the exponential decay rate for the second-moment estimates, and its literature value is 0.999. Both literature values work well with most datasets.
$$
\begin{aligned}
m_{t} = \beta_{1}m_{t-1}+(1-\beta_{1})g_t \\
v_t= \beta_{2}v_{t-1}+(1-\beta_2)g_t
\end{aligned}
$$
On a given iteration $t$, we can calculate the moving averages based on parameters $\beta_{1}, \beta_{2}$, and gradient $gt$. Since most algorithms that depend on moving averages such as SGD and RMSProp are biased, we need an extra step to correct the bias. This is known as the bias correction step:
$$
\begin{aligned}
\hat{m}_t=\frac{m_t}{1-\beta_{1}} \\
\hat{v}_t=\frac{v_t}{1-\beta_{2}}
\end{aligned}
$$
Finally, we can update the parameters (weights and biases) based on the calculated moving averages with a step size $\eta$:
$$
\begin{aligned}
W_t=W_{t-1}=\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

### Reference:
* [Medium: How to implement an Adam Optimizer from Scratch](https://medium.com/the-ml-practitioner/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc)