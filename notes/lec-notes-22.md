[toc]

---

<br><br>

## Lecture 21 Classification and Logistic Regression

### Deriving the Logistic Regression Model

<img src="lec-notes.assets/image-20220904210624652.png" alt="image-20220904210624652" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220904210727238.png" alt="image-20220904210727238" style="zoom:25%;" />

### The Logistic Regression Model

#### Sigmoid Function

<img src="lec-notes.assets/image-20220904211629978.png" alt="image-20220904211629978" style="zoom:25%;" />

#### From Feature to Probability

<img src="lec-notes.assets/image-20220904211925716.png" alt="image-20220904211925716" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220904212128767.png" alt="image-20220904212128767" style="zoom:25%;" />

#### Parameter Interppretation

<img src="lec-notes.assets/image-20220904213525185.png" alt="image-20220904213525185" style="zoom:25%;" />

### Parameter Estimation

<img src="lec-notes.assets/image-20220904215617562.png" alt="image-20220904215617562" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220904215934243.png" alt="image-20220904215934243" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220904220321264.png" alt="image-20220904220321264" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220904220348157.png" alt="image-20220904220348157" style="zoom:25%;" />

---

<br><br>

## Lecture 22 Logistic Regression II

### The Modeling Process

1) Choose a model
   - $\hat{y} = f_\theta(x) = x^T\theta$
2) Choose a loss function
   - Squares Loss or Absolute Loss
3) Fit the model
   - Regularization, Sklearn/Gradient descent
4) Evaluate model performance
   - $R^2$, Residuals, etc.


### Logistic Regression Model

```python
form sklearn.linear_model import LogisticRegression
model = LogisticRegression(fit_intercept=True)
model.fit(X, Y)
```

- optimal parameters:

    ```python
    model.intercept_, model.coef_
    ```

- predict the probabilities under the model:

    ```python
    model.predict_proba([[20]])		# 返回概率
    model.predict([[20]])           # 返回 1 或 0 (概率大于 0.5 则为 1)
    ```

- `.classes_` stores calss labels.

<img src="lec-notes.assets/image-20220905111209961.png" alt="image-20220905111209961" style="zoom:25%;" />

### Linear separability and Regularization

<img src="lec-notes-22.assets/image-20220905113735413.png" alt="image-20220905113735413" style="zoom:25%;" />

<img src="lec-notes-22.assets/image-20220905114608065.png" alt="image-20220905114608065" style="zoom:25%;" />

<img src="lec-notes-22.assets/image-20220905114838402.png" alt="image-20220905114838402" style="zoom:25%;" />

### Performance metrics

<img src="lec-notes-22.assets/image-20220905115550472.png" alt="image-20220905115550472" style="zoom:25%;" />

<img src="lec-notes-22.assets/image-20220905120237343.png" alt="image-20220905120237343" style="zoom:25%;" />

#### Precision and Recall

<img src="lec-notes-22.assets/image-20220905120935693.png" alt="image-20220905120935693" style="zoom:25%;" />

- **Precision** (also called positive predictive value) is the fraction of true positives among the total number of data points predicted as positive. (TP / predicted_P)h
- **Recall** (also known as sensitivity) is the fraction of true positives among the total number of data points with positive labels. (TP / total_P)<img src="images/../../images/lec-21-0.png" style="zoom: 25%;" > 

$$
\text{Precision} = \frac{n_\text{true\_positives}}{n_\text{true\_positives} + n_\text{false\_positives}}
$$

$$
\text{Recall} = \frac{n_\text{true\_positives}}{n_\text{true\_positives} + n_\text{false\_negatives}}
$$

### Adjusting the Classification Threshold

<img src="lec-notes-22.assets/image-20220905151215047.png" alt="image-20220905151215047" style="zoom:25%;" />

<img src="lec-notes-22.assets/image-20220905151543812.png" alt="image-20220905151543812" style="zoom:25%;" />

<img src="lec-notes-22.assets/image-20220905151621991.png" alt="image-20220905151621991" style="zoom:25%;" />

---

<br><br>
