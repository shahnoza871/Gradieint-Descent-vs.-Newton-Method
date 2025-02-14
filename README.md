# 🚀 Optimization via Gradient Descent & Newton’s Method

This project implements **Gradient Descent** and **Newton’s Method** with **backtracking line search** for unconstrained minimization. The program also handles **non-convex problems** by identifying and sorting the minimum points found by both methods, starting from different random initial points.

### 📊 Problem Formulation  
We minimize the function:  

\[
f(x, y) = \sin^2(x + y) + 5 + \cos^2(y) - e^{-\frac{x^2 + y^2 + 146y + 54x + 6058}{25}}
\]

---

## ⚡ Implementation Details

1. **Gradient Descent Method**  
   - Uses **backtracking line search** to dynamically adjust step size.  
   - Iteratively updates \( x, y \) based on the negative gradient.  
   - Stops when the gradient norm is below a threshold.  

2. **Newton’s Method**  
   - Uses the **Hessian matrix** for quadratic convergence.  
   - Backtracking ensures step size remains valid.  
   - Efficient for well-conditioned problems.  

3. **Comparison of Methods**  
   - The program starts from the same initial point and runs both optimization methods.  
   - Outputs the optimal points and function values for both approaches.  

---

## 📈 Results  
The program outputs:  
✅ **Optimal points** found by each method.  
✅ **Function values** at those points.  
✅ **Comparison** of results.  
