#!/usr/bin/env python
import cgi
import html
import numpy as np

from functions import *
from tabulation import *
from integration import integrate
from interpolation import interpolate
from cauchy import cauchy
from plots import *
from differentiation import *


def main():
    form = cgi.FieldStorage()

    print("Content-type: text/html\n")
    print("""<!DOCTYPE HTML>
            <html>
            <head>
                <link href="/css/bootstrap/css/bootstrap.min.css" rel="stylesheet">
                <script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>
                <script src="//ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
                <meta charset="utf-8">
                <title>Adds</title>
            </head>
            <body>""")

    print('<div class="container">')
    print('<legend>Differential equation system solution</legend>')


    pars = {
        'SA': None,
        'SB': None,
        'zA': None,
        'zB': None,
        'pA': None,
        'pB': None,
        'auto': None,
        'beta': None,
        'beta_1': None,
        'beta_2': None,
        'nbeta': None,
        'fA': None,
        'fB': None,
        'X0': None,
        'Y0': None,
        'T': None
    }

    # getting form inputs and checking them for numbers
    def get_par(par_name):
        par_value = html.escape(form.getfirst(par_name, "naan"))
        try:
            par_value = int(par_value)
            return par_value
        except:
            try:
                par_value = float(par_value)
                return par_value
            except:
                print("""<p>{} is not a number</p>
                        </body>
                        </html>""".format(par_name))
                return "NaN"

    for par_name in pars:
        if par_name not in ['auto', 'beta', 'beta_1', 'beta_2', 'nbeta']:
            par_value = get_par(par_name)
            if par_value == "NaN":
                return
            else:
                pars[par_name] = par_value
        else:
            if form.getvalue("auto_check"):
                pars['auto'] = True
            else:
                pars['auto'] = False

            
            if not pars['auto']:
                beta = get_par('beta')
                if beta == "NaN":
                    return
                else:
                    pars['beta'] = beta
            else:
                nbeta = get_par('nbeta')
                if nbeta == "NaN":
                    return
                else:
                    pars['nbeta'] = nbeta

                beta_1 = get_par('beta_1')
                if beta_1 == "NaN":
                    return
                else:
                    pars['beta_1'] = beta_1
                beta_2 = get_par('beta_2')
                if beta_2 == "NaN":
                    return
                else:
                    pars['beta_2'] = beta_2

    # special parameters sp_pars for deleting 'ones' where necessary
    sp_pars = {}
    for par_name, par_value in pars.items():
        sp_pars[par_name] = par_value
        if par_name in ['SA', 'SB', 'zA', 'zB', 'pA', 'fA', 'fB']:
            if par_value == 1:
                sp_pars[par_name] = ""

    SEGM_NUMB = 100
    MIN = 0
    MAX = pars['T']


    def do_beta(beta):

        # tabulating all functions
        tabulate(func_S, (pars['SA'], pars['SB']), SEGM_NUMB, 'func_S.txt', MIN, MAX)
        tabulate(func_z, (pars['zA'], pars['zB']), SEGM_NUMB, 'func_z.txt', MIN, MAX)
        tabulate(func_p, (pars['pA'], pars['pB']), SEGM_NUMB, 'func_p.txt', MIN, MAX)

        
        # tabulating function U(y)
        U_file = open('func_U.txt', 'w')
        y = MIN
        for n in range(SEGM_NUMB + 1):
            integral = integrate('func_p.txt', y, 1)
            y += (MAX - MIN) / SEGM_NUMB

            U_file.write(str(y) + ' ' + str(integral) + '\n')
        U_file.close()

        # interpolating function U(y)
        U_coef = interpolate('func_U.txt', 'coef_func_U.txt')

        # interpolating function z(t)
        z_coef = interpolate('func_z.txt', 'coef_func_z.txt')
        der_z_coef = derivative(z_coef)

        #interpolating function S(t)
        S_coef = interpolate('func_S.txt', 'coef_func_S.txt')

        #interpolating function p(w)
        p_coef = interpolate('func_p.txt', 'coef_func_p.txt')


        # first right part of the system of differential equations
        def rp1(t, x, y):
            return cube_func(t, der_z_coef, MIN, MAX) * cube_func(y, U_coef, MIN, MAX)

        # second right part of the system of differential equations
        def rp2(t, x, y):
            return beta * (pars['fA'] * cube_func(t, S_coef, MIN, MAX) - pars['fB'] * x)

        # solving system of differential equations
        try:
            x, y = cauchy(rp1, rp2, pars['X0'], pars['Y0'], MAX, SEGM_NUMB)
        except:
            print("<h4 style='color:red'>Too big numbers :( Try a little bit less.</h4>")

        t = np.linspace(MIN, MAX, SEGM_NUMB + 1)
        x_file = open('func_x.txt', 'w')
        y_file = open('func_y.txt', 'w')
        for i, ti in enumerate(t):
            x_file.write(str(ti) + ' ' + str(x[i]) + '\n')
            y_file.write(str(ti) + ' ' + str(y[i]) + '\n')
        x_file.close()
        y_file.close()


        # making function x(t) - S(t)
        x_file = open('func_x.txt', 'r')
        S_file = open('func_S.txt', 'r')
        xS_file = open('func_xS.txt', 'w')
        for i in range(SEGM_NUMB + 1):
            x_file_line = x_file.readline()
            xS_file.write(x_file_line.split()[0] + ' ')
            xS_file.write(str(float(x_file_line.split()[1]) - float(S_file.readline().split()[1])) + '\n')
        x_file.close()
        S_file.close()
        xS_file.close()


        # quality criteria

        # tabulating function w*p(w)
        def func_wp(w):
            return w*cube_func(w, p_coef, MIN, MAX)
        tabulate(func_wp, (), SEGM_NUMB, 'func_wp.txt', MIN, MAX)

        # interpolating derivative of function x(t)
        x_coef = interpolate('func_x.txt', 'coef_func_x.txt')
        der_x_coef = derivative(x_coef)

        # tabulating function V(y) = x'(t) * int{y}{1}w*p(w)*dw
        V_file = open('func_V.txt', 'w')
        t = MIN
        for n in range(SEGM_NUMB + 1):
            result = cube_func(t, der_x_coef, MIN, MAX) * integrate('func_wp.txt', y[n], 1)
            t += (MAX - MIN) / SEGM_NUMB

            V_file.write(str(t) + ' ' + str(result) + '\n')
        V_file.close()

        C1 = 1 - integrate('func_V.txt', 0, MAX) / (x[-1] - pars['X0'])
        C2 = np.abs(x[-1] - cube_func(MAX, S_coef, MIN, MAX)) / cube_func(MAX, S_coef, MIN, MAX)

        return C1, C2



    print('<nav class="navbar">')
    print('<ul class="nav nav-pills">')

    print('<li class="nav-item" id="gr1">')
    print('<a class="nav-link active" lang="latex">\\rho (\\omega)</a>')
    print('</li>')
    print('<li class="nav-item" id="gr2">')
    print('<a class="nav-link" lang="latex">x(t)</a>')
    print('</li>')
    print('<li class="nav-item" id="gr3">')
    print('<a class="nav-link" lang="latex">S(t)</a>')
    print('</li>')
    print('<li class="nav-item" id="gr4">')
    print('<a class="nav-link" lang="latex">z(t)</a>')
    print('</li>')
    print('<li class="nav-item" id="gr5">')
    print('<a class="nav-link" lang="latex">x(t) - S(t)</a>')
    print('</li>')
    print('<li class="nav-item" id="gr6">')
    print('<a class="nav-link" lang="latex">y(t)</a>')
    print('</li>')

    if pars['auto']:
        print('<li class="nav-item" id="gr7">')
        print('<a class="nav-link" lang="latex">F(\\beta)</a>')
        print('</li>')

    print('</ul>')
    print('</nav>')

    plot('func_p.txt', 'func_p', 'omega', 'rho', MIN, MAX)
    print("<img src='/plot_func_p.png' id='im1' width='80%' height='70%'>")
    plot('func_x.txt', 'func_x', 't', 'x', MIN, MAX)
    print("<img src='/plot_func_x.png' id='im2' width='80%' height='70%'>")
    plot('func_S.txt', 'func_S', 't', 'S', MIN, MAX)
    print("<img src='/plot_func_S.png' id='im3' width='80%' height='70%'>")
    plot('func_z.txt', 'func_z', 't', 'z', MIN, MAX)
    print("<img src='/plot_func_z.png' id='im4' width='80%' height='70%'>")
    plot('func_xS.txt', 'func_xS', 't', 'x - S', MIN, MAX)
    print("<img src='/plot_func_xS.png' id='im5' width='80%' height='70%'>")
    plot('func_y.txt', 'func_y', 't', 'y', MIN, MAX)
    print("<img src='/plot_func_y.png' id='im6' width='80%' height='70%'>")

    if pars['auto']:
        plot('func_F.txt', 'func_F', 'beta', 'F', pars['beta_1'], pars['beta_2'])
        print("<img src='/plot_func_F.png' id='im7' width='80%' height='70%'>")
    

    if not pars['auto']:
        C1, C2 = do_beta(pars['beta'])
    else:
        FA = 1
        FB = 10
        betas = []
        Fbetas = []
        b = pars['beta_1']
        for i in range(pars['nbeta'] + 1):
            C1i, C2i = do_beta(b)
            betas.append(b)
            Fbetas.append(FA*C1i + FB*C2i)
            b += (pars['beta_2'] - pars['beta_1']) / pars['nbeta']

        # tabulating function F
        F_file = open('func_F.txt', 'w')
        for i in range(len(betas)):
            F_file.write(str(betas[i]) + ' ' + str(Fbetas[i]) + '\n')
        F_file.close()

        best_arg = np.argmin(Fbetas)
        C1, C2 = do_beta(betas[best_arg])

        print("<h4>Criteria:</h4>")
        print("<p><div lang='latex'>\\\\ Best F = {} \\\\ Best \\beta = {}</div></p>".format(Fbetas[best_arg], betas[best_arg]))



    if not pars['auto']:
        print("<h4>Criteria:</h4>")
    print("<p><div lang='latex'>\\\\ C_1 = {} \\\\ C_2 = {}</div></p>".format(C1, C2))

    # printing system of differential equations
    print('<h4 id="qq1">Your system of differential equations:</h4>')
    print("<p><div lang='latex'> \
        \\\\ \\frac{{dx}}{{dt}} = z'(t) \\int_{{y}}^{{1}} \\rho (\\omega) d \\omega \
        \\\\ \\frac{{dy}}{{dt}} = \\beta ({} S(t) - {} x) \
        \\\\ z(t) = {} t + {} cos(t) \
        \\\\ \\rho (\\omega) = {} \\omega ({} - \\omega) \
        \\\\ S(t) = {} t + {} sin(t) \
        \\\\ x(t = 0) = {} \
        \\\\ y(t = 0) = {} \
        \\\\ T = {}".format(sp_pars['fA'], sp_pars['fB'], sp_pars['zA'], sp_pars['zB'], sp_pars['pA'],
            sp_pars['pB'], sp_pars['SA'], sp_pars['SB'], sp_pars['X0'], sp_pars['Y0'], sp_pars['T']))

    if not pars['auto']:
        print("\\\\ \\beta = {}</div></p>".format(sp_pars['beta']))
    else:
        print("\\\\ \\beta = {} .. {}, nbeta = {}</div></p>".format(sp_pars['beta_1'], sp_pars['beta_2'], sp_pars['nbeta']))


    print("""<script>
                function hideevth() {
                    for (var i = 1; i <= 7; ++i) {
                        $('#im' + i.toString()).hide();
                    }
                }

                function clear_color() {
                    for (var i = 1; i <= 7; ++i) {
                        $('#gr' + i.toString()).css('background-color', '#FFFFFF');
                    }
                }

                hideevth();
                $('#im1').show();
                $('#gr1').css('background-color', '#EEEEEE');

                $('.nav-item').click(function() {
                    hideevth();
                    var i = $(this).attr('id').toString()[2];
                    $('#im' + i).show();
                    clear_color();
                    $('#gr' + i).css('background-color', '#EEEEEE');
                });
            </script>
            """)


                

    print('</body></html>')

if  __name__ ==  "__main__" :
    main()