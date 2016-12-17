
# coding: utf-8

# ### Решение систем линейных алгебраических уравнений
# 
# #### Метод Гаусса с выбором главного элемента по столбцу
# 
# Присвоем номер 1 тому уравнению, в котором коэффициент при $x_1$ наибольший по модулю. Этот коэффициент отличен от нуля, так как противное означало бы, что матрица $A$ имеет нулевой первый столбец, то есть вырождена. После этой перенумерации уравнений сделаем первый шаг метода Гаусса:
# 
# Поделим первое уравнение системы на $a_{11}$; $(n - 1)$ раз умножим это уравнение на $a_{i1}$ и вычтем его из $i-го$ уравнения системы.
# 
# Затем этот же процесс применяется к подматрице $A^{(1)} = (a_{ij}^{(1)})_{i,j=2,...,n} \in M_{n - 1}$ и так далее.
# 
# Применимость: ко всем невырожденным матрицам.
# 
# Главный элемент выбирается для того, чтобы уменьшить вычислительную погрешность. Погрешность будет наименьшей, если модуль отношения ${a_{i1}}/{a_{11}}$ наименьший из возможных.
# 
# Ассимптотическое количество операций: $\frac{2}{3} n^3 + O(n^2)$

# In[1]:

import numpy as np

def sle_gauss(A, b):
    """
    Solves system of linear equations: A*x=b
    
    Args:
        A (numpy.ndarray)
        b (numpy.ndarray)
    
    Returns:
        x (numpy.ndarray)
    """
    X = np.hstack((A, np.split(b, len(b)))).astype(float)

    # Forward Elimination
    for step in range(len(X)):
        maxi = np.argmax(X[step:, step]) + step
        X[step], X[maxi] = X[maxi].copy(), X[step].copy()
        
        X[step] = X[step] / X[step][step]

        for i in range(step + 1, len(X)):
            X[i] -= X[step] * X[i][step]
        
    
    # Back Substitution
    for i in range(len(X) - 2, -1, -1):
        for j in range(i + 1, len(X)):
            X[i] -= X[j] * X[i, j]

    return X[:, -1]


# In[2]:

A = np.asarray([[1, 2, 3], [2.001, 3.999, 6], [15, 3, 6]])
print('A:', A)
b = np.asarray([1, 2, 3])
print('b:', b)

print('Определитель:', np.linalg.det(A))

print('Бесконенчая норма:', np.linalg.norm(A, np.inf))

print('Число обусловленности:', np.linalg.cond(A, np.inf))

x = sle_gauss(A, b)
print('Решение:', x)

print('Модуль невязки:', np.linalg.norm(A.dot(x) - b))


# In[3]:

from scipy.linalg import hilbert

A = hilbert(8)
print('A:', A)
b = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
print('b:', b)

print('Определитель:', np.linalg.det(A))

print('Бесконенчая норма:', np.linalg.norm(A, np.inf))

print('Число обусловленности:', np.linalg.cond(A, np.inf))

x = sle_gauss(A, b)
print('Решение:', x)

print('Модуль невязки:', np.linalg.norm(A.dot(x) - b))


# In[4]:

A = np.asarray([[10**6, 2], [10**13, 2]])
print('A:', A)
b = np.asarray([1, 2])
print('b:', b)

print('Определитель:', np.linalg.det(A))

print('Бесконенчая норма:', np.linalg.norm(A, np.inf))

print('Число обусловленности:', np.linalg.cond(A, np.inf))

x = sle_gauss(A, b)
print('Решение:', x)

print('Модуль невязки:', np.linalg.norm(A.dot(x) - b))


# ### Интерполирование
# 
# #### Интерполирование кубическим сплайном дефекта 1
# 
# Методом Гаусса решается система линейных уравнений:
# 
# $$
# \begin{bmatrix}
#     \frac{\tau_0 + \tau_1}{3} & \frac{\tau_1}{6} & 0 & \dots & 0 \\
#     \frac{\tau_1}{6} & \frac{\tau_1 + \tau_2}{3} & \frac{\tau_2}{6} & \dots & 0 \\
#     \dots & \dots & \dots & \dots & \dots\\
#     \dots & \dots & \dots & \dots & \dots\\
#     \dots & \dots & \dots & \dots & \dots\\
# \end{bmatrix}
# \begin{bmatrix}
#     m_1 \\
#     m_2 \\
#     \dots \\
#     \dots \\
#     m_{N - 1} \\
# \end{bmatrix}
# =
# \begin{bmatrix}
#     \frac{f_2 - f_1}{\tau_1} - \frac{f_1 - f_0}{\tau_0} \\
#     \dots \\
#     \frac{f_{n + 1} - f_n}{\tau_n} - \frac{f_n - f_{n - 1}}{\tau_{n - 1}} \\
#     \dots \\
#     \frac{f_N - f_{N - 1}}{\tau_{N - 1}} - \frac{f_{N - 1} - f_{N - 2}}{\tau_{N - 2}} \\
# \end{bmatrix}
# $$
# 
# где $ \tau_n = t_{n + 1} - t_n $
# 
# Далее вычисляются:
# 
# $
# A_n = \frac{f_{n + 1} - f_n}{\tau_n} - \frac{\tau_n}{6} (m_{n + 1} - m_n)
# $
# 
# $
# B_n = f_n - \frac{1}{6} m_n \tau_n ^2 - A_n t_n
# $
# 
# Итоговый кубический многочлен для каждого $ n \in \{ 0, N - 1 \} $ вычисляется по формуле:
# 
# $
# S_n (t) = \frac{1}{6 \tau_n} (m_n (t_{n + 1} - t)^3 + m_{n + 1} (t - t_n)^3 ) + A_n t + B_n
# $

# #### Примеры функций

# In[5]:

import numbers

# гладкая функция
def s_func1(x):
    return 9*x - 3*(x**2)

# разрывная функция
def d_func1(x):
    if isinstance(x, numbers.Number):
        if x < 0.5:
            return x / 2
        else:
            return x / 2 + 0.5
    else:
        ans = []
        for xi in x:
            if xi < 0.5:
                ans.append(xi / 2)
            else:
                ans.append(xi / 2 + 0.5)
        return np.asarray(ans)

# осциллирующая функция
def o_func1(x):
    return x*np.cos(100*x)


# #### Табулирование

# In[6]:

def tabulate(func, args, segm_numb, file_name):
    f = open(file_name, 'w')
    x = 0
    n = 0
    while n <= segm_numb:
        f.write(str(x) + ' ' + str(func(x, *args)) + '\n')
        x += 1 / segm_numb
        n += 1
    f.close()


# #### Функция интерполирования
# 
# Считывает значения из файла func_file_name
# 
# Возвращает numpy.ndarray размера $ N * 4 $, где в каждой строке стоят 4 коэффициента соответствующего многочлена $S_n$

# In[7]:

def spline_interpolation(func_file_name):
    func_file = open(func_file_name, 'r')
    func = []
    for line in func_file:
        x, y = line.split()
        func.append((float(x), float(y)))
    func_file.close()
    
    N = len(func) - 1
    
    tau = np.zeros(N)
    for i in range(N):
        tau[i] = func[i + 1][0] - func[i][0]
    
    T = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        T[i][i] = (tau[i] + tau[i + 1]) / 3
    for i in range(N - 2):
        T[i + 1][i] = (tau[i + 1]) / 6
        T[i][i + 1] = (tau[i + 1]) / 6
    
    F = np.zeros(N - 1)
    for i in range(N - 1):
        F[i] = (func[i + 2][1] - func[i + 1][1]) / tau[i + 1] - (func[i + 1][1] - func[i][1]) / tau[i]
    
    M = sle_gauss(T, F)
    M = np.hstack((np.asarray([0]), M, np.asarray([0])))
    
    A = np.zeros(N)
    B = np.zeros(N)
    for i in range(N):
        A[i] = (func[i + 1][1] - func[i][1]) / tau[i] - tau[i] / 6 * (M[i + 1] - M[i])
        B[i] = func[i][1] - M[i] * tau[i]**2 / 6 - A[i] * func[i][0]
    
    C = np.zeros((N, 4))
    for i in range(N):
        C[i][0] = (M[i] * func[i + 1][0]**3 - M[i + 1] * func[i][0]**3) / tau[i] / 6 + B[i]
        C[i][1] = (3 * func[i][0]**2 * M[i + 1] - 3 * func[i + 1][0]**2 * M[i]) / tau[i] / 6 + A[i]
        C[i][2] = (-3 * func[i][0] * M[i + 1] + 3 * func[i + 1][0] * M[i]) / tau[i] / 6
        C[i][3] = (-M[i] + M[i + 1]) / tau[i] / 6
    
    return C


# In[8]:

def cube_func(x, coef):
    return coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3


# #### По матрице коэффициентов С возвращает табулированную функцию

# In[9]:

def get_interpolated_func(C, step=10):
    func = []
    x = 0
    for c in C:
        linspace = np.linspace(x, x + 1/len(C), step, endpoint=False)
        for i in range(step):
            f = cube_func(linspace, c)
            func.append(f[i])
        x += 1 / len(C)
    return func


# In[10]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# #### Рисует график функции, сплайн с сеткой размера step_numb, график ошибки

# In[11]:

def plot_interpolation(true_func, step_numb=100, step=10, error_plot=False):
    tabulate(true_func, (), step_numb, 'func.txt')
    
    linspace = np.linspace(0, 1, step_numb * step)
    plt.plot(linspace, true_func(linspace), color='black', alpha=0.5, label='f(x)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolation')
    
    C = spline_interpolation('func.txt')
    interp_func = get_interpolated_func(C, step)
    
    linspace = np.linspace(0, 1, step_numb * step)
    plt.plot(linspace, interp_func, color='red', alpha=0.5, label='Spline')
    plt.legend()
    plt.show()
    
    if error_plot:
        plt.plot(linspace, true_func(linspace) - interp_func, color='red')
        plt.title('Error plot: f(x) - S(x)')


# #### Гладкая функция

# In[12]:

plot_interpolation(s_func1, step_numb=5)


# In[13]:

plot_interpolation(s_func1, step_numb=50, error_plot=True)


# Видно, что для гладкой функции достаточно не густой сетки для хорошего приближения

# #### Разрывная функция

# In[14]:

plot_interpolation(d_func1, step_numb=10)


# In[15]:

plot_interpolation(d_func1, step_numb=100, error_plot=True)


# Для разрывной функции плохо приближает в месте разрыва, для густой сетки от 100 разбиений приближение хорошее
# 
# На графике ошибки видно, что ошибка везде нулевая, кроме места разрыва

# #### Осциллирующая функция

# In[16]:

plot_interpolation(o_func1, step_numb=30)


# In[17]:

plot_interpolation(o_func1, step_numb=50, error_plot=True)


# Для осцилирующей функции хорошее приближение для густый сеток от 50 разбиений
# 
# На графике ошибки видно, что ошибка увеличивается при увеличении аргумента
