# The goal is to investigate different possibilities
# to fit a non-linear model using Linear Regression.

#===================================================
# Author: Marjan Khamesian
# Date: June 2020
#===================================================

import numpy as np
import matplotlib.pyplot as pl

# === Generating a 'non-linear' dataset ======================
# Adding some randomness to y as well with the same shape of x
x = 6*np.random.random(300)
a = 0.1
b = 0.03
c = 1.5

y = a*x + b*(x**3) + c*np.sin(x) + np.random.random(x.shape[0])

# === Plot of the generated data ==========
fig, ax = pl.subplots(1,1, figsize = (8,6))
ax.plot(x,y,'b.', linewidth = 3)
#pl.show()

# x array is required to be two-dimensional,
# with one column and as many rows as necessary.  
X = x.reshape(-1,1)

# ---------- Create a model and fit it ----------
# ===============================================
# Linear regression method from `scikit-learn`
# =============================================== 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()  

# Fitting the data
lm.fit(X,y)

# ----------------------- Results ---------------------------------
# ==== Coefficient of determination (R**2) ========================

# to check whether the model works satisfactorily and interpret it.
r_sq = lm.score(X, y)
print('coefficient of determination:', r_sq)

# === coeff b_0 
print('intercept:', lm.intercept_)

# === coeff b_1
print('slope:', lm.coef_)


# ------------------------------ 
# === Prediction =============== 

# Predicting 'y' after learning. 
y_pred = lm.predict(X)

# ------------------------------
# === Plot =====================
pl.plot(x,y_pred, 'g.')
pl.plot(x,y,'b.', linewidth = 3)
#pl.show()

# -------------------------------------------
# === Residual ==============================
# Plot the distribution of the residual:
# difference of observed and predicted values 
del_y = y - y_pred
pl.hist(del_y, bins=40)
#pl.show()

# ===========================
# Higher Polynomial Expansion 
# ===========================

X_p = np.c_[x,x*x, x**3, x**4]
print("Dimension of X: ", X_p.shape)

# Model
lm2 = LinearRegression()

# Fitting
lm2.fit(X_p,y)

# Coefficient of determination                                                  
r_sq = lm2.score(X_p, y)
print('coefficient of determination 2:', r_sq)

# Prediction
y_pred1= lm2.predict(X_p)

# Plotting
pl.plot(x,y_pred1, 'g.')
pl.plot(x,y,'b.', linewidth = 3, alpha = 0.2)
#pl.show()

# ======================
# Problem of overfitting
# ======================

# The model that we built is perfect, but we want to see if it can
# have the same performance on `unseen` data.

# ===============================================================
# Exploring the different form of expansion for a better fitting. 
#================================================================

# === Read train-test split method ==============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_p, y, test_size = 0.3)

# === Model ============
lm2 = LinearRegression()

# === Fitting ==========
lm2.fit(X_train,y_train)

# === Coefficient of determination ===========
r_sq = lm2.score(X_train, y_train)
print('coefficient of determination 3:', r_sq)

# === Prediction ===========
y_pred1= lm2.predict(X_test)

# === Plotting ================================
pl.plot(X_test[:,0],y_pred1, 'g*', linewidth=4)
pl.plot(x,y,'b.', linewidth = 1, alpha = 0.2)
#pl.show()

# === Residual =========
del_y = y_test - y_pred1
pl.hist(del_y, bins=40)
pl.show()









