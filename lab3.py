import numpy as np
import random

def f(x, y):
    return (np.sin(x + y))**2 + 5 + (np.cos(y))**2 - np.exp(-(x**2 + y**2 + 146*y + 54*x + 6048) / 25)

def gradient_f(x, y):
    df_dx = 2 * np.sin(x + y) * np.cos(x + y) - (2 * x + 54) * np.exp(-(x**2 + y**2 + 146*y + 54*x + 6048) / 25) / 25
    df_dy = 2 * np.sin(x + y) * np.cos(x + y) - (2 * y + 146) * np.exp(-(x**2 + y**2 + 146*y + 54*x + 6048) / 25) / 25 + 2*np.cos(y)
    return [df_dx, df_dy]

def gradlength(x, y):
    return gradient_f(x, y)[0]**2+gradient_f(x, y)[1]**2

def backtracking(x, y, steplength, alpha=0.1, betta=0.5):
    if f(x-steplength*gradient_f(x, y)[0], y-steplength*gradient_f(x, y)[1]) > f(x, y) - alpha*steplength*gradlength(x, y):
        return steplength*betta
    return steplength                       

def gradient_decent(init_var, steplength, error):
    points = []
    points.append(init_var)
    gradient = gradient_f(points[-1][0], points[-1][1])
    
    step = 0
    # while abs(gradient[0])>error and abs(gradient[1])>error:
    while np.sqrt(gradlength(points[-1][0], points[-1][1]))>error:
        points.append([points[-1][0]-gradient[0]*steplength, points[-1][1]-gradient[1]*steplength])
        gradient = gradient_f(points[-1][0], points[-1][1])
        if step == 100:
            # print('FORCED BREAK')
            break
        steplength=backtracking(points[-1][0], points[-1][1], steplength)
        step+=1
    return points[-1][0], points[-1][1]


def hessian_f(x, y):
    exp_term=np.exp(-(x**2+y**2+146*y+54*x+6048)/25)
    df2_dx2=-2*np.sin(x+y)**2+2*np.cos(x+y)**2-exp_term*(2-(2*x+54)**2/25)
    df2_dy2=-2*np.sin(x+y)**2+2*np.cos(x+y)**2-exp_term*(2-(2*y+146)**2/25)
    df2_dxdy=2*np.cos(x+y)**2-exp_term*(2*x+54)*(2*y+146)/25

    return np.array([[df2_dx2, df2_dxdy], [df2_dxdy, df2_dy2]])


def newton_method(init_var, error, max_steps=100):
    x, y = init_var
    gradient = np.array(gradient_f(x, y))
    step = 0

    while np.linalg.norm(gradient) > error:
        hessian = hessian_f(x, y) + np.eye(2) * 1e-5
        hessian_inv = np.linalg.inv(hessian)
        update = hessian_inv.dot(gradient)
        
        alpha = 1.0
        # alpha = backtracking(x, y, steplength=1)
        while f(x - alpha * update[0], y - alpha * update[1]) > f(x, y):
            alpha *= 0.5
            if alpha < 1e-8:
                print("Step size too small. Terminating.")
                return x, y
        
        x -= alpha * update[0]
        y -= alpha * update[1]
        gradient = np.array(gradient_f(x, y))
        step += 1
        
        if step >= max_steps:
            break
    
    return x, y


best_x=-25
best_y=-75
x, y = gradient_decent([best_x, best_y],  steplength=5, error=10^(-3))
print(f"Best optimal by Gradient Descend :{x, y}")
print(f"Value of the function: {f(x, y)}")
x, y = newton_method([best_x, best_y], error=10^(-3))
print(f"Best optimal by Newton's Method : {x, y}")
print(f"Value of the function: {f(x, y)}")


