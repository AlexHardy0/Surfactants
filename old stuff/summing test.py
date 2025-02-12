import sympy as sp

# Define symbols
kbt, gamma = sp.symbols("kBT gamma")

# Define indexed variables
c = sp.IndexedBase('c')
mu = sp.IndexedBase('mu')

# Define index i
i = sp.Idx('i')

# Define the expression
expr1 =  (1/gamma)*(c[i+1]*(mu[i+2]-mu[i]) + c[i-1]*(mu[i]-mu[i-2]))

expr2 = (kbt/gamma)*(c[i+1] + c[i-1] - 2*c[i])

# Perform summation from i=-1 to 100
sum_expr = sp.summation(expr1, (i, -1, 10))

# Output the result
print(sp.simplify(sum_expr))


