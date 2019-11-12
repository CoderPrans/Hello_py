import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

def getB_i(x, y, m_x, m_y):
    # number of observations/points
    n = np.size(x)
    #calculating cross-deviation and deviation about xi
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    #calculating regression coefficients
    return SS_xy / SS_xx

def estimate_coef(x,y):
    #mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    b_1 = getB_i(x, y, m_x, m_y)
    b_0 = m_y - b_1*m_x

    print("b_1:")
    print(b_1)
    print("b_0:")
    print(b_0)
    print("m_x:")
    print(m_x)
    print("m_y:")
    print(m_y)

    return(b_0, b_1)

def plot_regression_line(x, y, b, xlabel, ylabel):
    pt.title("Linear Regression") 
    # plotting the actual points as scatter plot
    pt.scatter(x, y, color = "m", marker = "o", s = 30)
    # predicted response vector
    y_pred = b[0] + b[1]*x
    # plotting the regression line
    pt.plot(x, y_pred, color="g")
    # putting labels
    pt.xlabel(xlabel)
    pt.ylabel(ylabel)

    # function to show plot
    pt.show()

def genVariables(columns, df):
    variables = {}
    for xi in columns:
        variables[xi] = df[xi].to_numpy()
    return variables

def doRegression(**args):
    i = 2
    x = args['x'][i]
    y = args['y']
    # get labels
    xlabel = args['cols'][i]
    ylabel = args['cols'][-1]
    # estimating coefficients
    b = estimate_coef(x, y)
    # plotting regression line
    plot_regression_line(x, y, b, xlabel, ylabel)


if __name__ == "__main__":

    df = pd.read_csv('octane_rating.txt', comment="#")
    columns = df.columns.to_numpy()
    
    variables = genVariables(columns, df)

    x0 = variables['Material 1 amount']
    x1 = variables['Material 2 amount']
    x2 = variables['Material 3 amount']
    x3 = variables['Condition']
    x4 = variables['Octane number']

    doRegression(x=[x0, x1, x2, x3], y=x4, cols=columns[2:])
   
