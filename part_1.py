import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.integrate


A = 2
w = 3
B = 1
n = 12


#defining target function
def target(x):
    return A*scipy.sin(w*x)+B


#universal approximation function
def approx_decorator(coeffs,system):
    p=len(coeffs)
    return lambda x: sum(coeffs[i]*system[i](x) for i in range(p))


#calculating error using definition in L2
def error(target, approx):
    return integrate.quad(lambda x: (target(x)-approx(x))**2,-scipy.pi,scipy.pi)[0]


def plot_result(approx):
    t=np.linspace(-np.pi,np.pi,1000)
    #plotting target and its approx wih linear independant system
    approx_val=[approx(time) for time in t]
    target_val=[target(time) for time in t]
    plt.plot(t,approx_val)
    plt.plot(t,target_val)
    plt.axis([-np.pi,np.pi,-10,10])
    plt.show()


def product(f,g):
    return integrate.quad(lambda x: f(x)*g(x), -scipy.pi, scipy.pi)[0]


def main_1 ():
    # filling in Gramms matrix
    G = [[integrate.quad(lambda x: scipy.exp((i + j) * x), -scipy.pi, scipy.pi)[0] for j in range(n + 1)] for i in
         range(n + 1)]
    G = np.array(G)
    #print(G)

    # filling in right hand-side of matrix equation
    F = [integrate.quad(lambda x: scipy.exp(i * x) * target(x), -scipy.pi, scipy.pi)[0] for i in range(n + 1)]
    F = np.array(F)
    #print(F)

    # calculating coef of approx func
    indep_coef = np.linalg.solve(G, F)
    #print("coef: ", indep_coef)

    # linear independent system
    phi = [(lambda y: (lambda x: scipy.exp(y * x)))(i) for i in range(n + 1)]


    print("error for esp(kx): ",error(target, approx_decorator(indep_coef, phi)))

    # plotting result for approx with linear independent system
    plot_result(approx_decorator(indep_coef, phi))

    # approximation with trigonometric system
    psi = []
    psi.append(lambda x: 1 / scipy.sqrt(2 * scipy.pi))

    for i in range(1, n + 1):
        psi.append(lambda x, i=i: 1 / scipy.sqrt(scipy.pi) * scipy.sin(i * x))
        psi.append(lambda x, i=i: 1 / scipy.sqrt(scipy.pi) * scipy.cos(i * x))

    # we can omit norm in denominator as trig system is orthonormed
    trig_coef = [product(target, psi[i]) for i in range(2 * n + 1)]
    print(trig_coef)

    print("error for trig ", error(target, approx_decorator(trig_coef, psi)))
    # plotting approx with trig system
    plot_result(approx_decorator(trig_coef, psi))

#main_1()