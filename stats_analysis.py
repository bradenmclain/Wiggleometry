import pandas as pd
import statsmodels.api as sm


x = [99.34,97.94,88.51,14.81,27.14,91.78,10.07,1.38,4.31,23.81,0.00,7.28,15.49,6.25,8.78,0.00,8.57,2.16,-23.47,0.00,0.00,0.00,-100.0,0.00,-100.0,-100.0,-100.0]
y = [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,2,0,0,0,2,0,2,2,2]

print(len(x))
print(len(y))



X = sm.add_constant(x)  # Adds an intercept term to the model
model = sm.GLM(y, x)
result = model.fit()
print(result.summary())
print(result.stats.anova)