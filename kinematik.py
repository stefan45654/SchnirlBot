import numpy as np
from scipy.optimize import least_squares
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt


def kin(params, x, y, xm2, l1, l2, l3, plot):
    """x = a[0];
	y = a[1];
	xm2 = a[2];
	l1 = a[3];
	l2 = a[4];
	l3 = a[5];"""

    s1 = params[0]
    alpha = params[1]
    gamma = params[2]


    pos = np.array([[x], [y], [0]])
    vec_0M2_i = np.array([[xm2], [0], [0]])
    vec_0A_s1 = np.array([[s1], [0], [0]])
    vec_AB_d = np.array([[l1], [0], [0]])
    vec_ASP_d = np.array([[l1 / 2], [-l2], [0]])
    vec_AP_d = np.array([[l1 / 2], [-l3], [0]])

    trans_is1 = np.array([[np.cos(-alpha), -np.sin(-alpha), 0], [np.sin(-alpha), np.cos(-alpha), 0], [0, 0, 1]])
    trans_id = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    vec_0A_i = np.matmul(trans_is1, vec_0A_s1)
    vec_AB_i = np.matmul(trans_id,vec_AB_d)
    vec_0B_i = vec_0A_i+vec_AB_i
    vec_BM2_i = vec_0M2_i-vec_0B_i
    vec_M2B_i = -vec_BM2_i
    s2 = np.sqrt(np.power(vec_BM2_i[0][0],2)+np.power(vec_BM2_i[1][0],2))

    vec_ASP_i = np.matmul(trans_id, vec_ASP_d)
    vec_AP_i = np.matmul(trans_id, vec_AP_d)

    """calculate intersection of M1A and M2B"""
    mat = np.block([[vec_0A_i, -vec_M2B_i, np.array([[0], [0], [1]])]])
    try:
        para = np.matmul(inv(mat), vec_0M2_i)
        inter = vec_0A_i * para[0][0]
        vec_0SP_i = vec_0A_i + vec_ASP_i
        res1 = vec_0SP_i[0][0] - inter[0][0]
    except:
        res1 = 100

    """make the position of the pen match x and y"""
    vec_0P_i = vec_0A_i + vec_AP_i

    res2 = vec_0P_i[0][0] - x
    res3 = vec_0P_i[1][0] - y

    vec_0B_i = vec_0A_i+vec_AB_i;
    if plot ==1:
        plt.plot((0,vec_0A_i[0][0]),(0,vec_0A_i[1][0]), '-k')
        plt.plot((vec_0A_i[0][0],vec_0B_i[0][0]),(vec_0A_i[1][0],vec_0B_i[1][0]),'-b')
        plt.plot((vec_0M2_i[0][0],vec_0M2_i[0][0]-vec_BM2_i[0][0]),(vec_0M2_i[1][0],vec_0M2_i[1][0]-vec_BM2_i[1][0]), '-k')
        plt.plot((vec_0A_i[0][0],inter[0][0],vec_0M2_i[0][0]-vec_BM2_i[0][0]),(vec_0A_i[1][0],inter[1][0],vec_0M2_i[1][0]-vec_BM2_i[1][0]),'-r')
        plt.plot((vec_0P_i[0][0]),(vec_0P_i[1][0]),'xk')
        plt.plot((vec_0SP_i[0][0]),(vec_0SP_i[1][0]), 'xr')
        plt.plot((x), (y), 'ob')
        plt.show()


    return (res1,res2,res3)


"""a = (x,y,xm2,l1,l2,l3)"""
"""params = (s1,s2,alpha,beta,gamma)"""

f = open('../../../gcode_inkscape_tests/test.gcode', 'r')
f_new = open('../../../gcode_inkscape_tests/test_rewrite.gcode', 'w')
last_pos = np.array([[0], [0]])
x = 0
y = 0
z = 0
feed = 100
g01_step_size = 30
g00_step_size = 10000
XM2 = 1564-15.5
L1 = 60
L2 = 130/2
L3 = 20
H1 = 2000
dist_AP = np.sqrt(np.power(L1/2,2)+np.power(L3,2))

s1_plot = np.array([])
s2_plot = np.array([])
s10_plot = np.array([])
s20_plot = np.array([])

for line in f:
    l = (line.split(';')[0]).rstrip('\n').split(' ')
    command = l[0]
    if command == 'G00' or command == 'G01':
        for ele in l:
            if ele[0] == 'X':
                x = float(ele.lstrip('X'))
            elif ele[0] == 'Y':
                y = float(ele.lstrip('Y'))
            elif ele[0] == 'Z':
                z = float(ele.lstrip('Z'))
            elif ele[0] == 'F':
                feed = float(ele.lstrip('F'))
        if command == 'G00':
            step_size = g00_step_size
        else:
            step_size = g01_step_size
        new_pos = np.array([[x], [y]])
        vec = new_pos - last_pos
        dist = np.sqrt(np.power(vec[0][0], 2) + np.power(vec[1][0], 2))
        steps = np.arange(step_size, dist, step_size)
        steps = np.block([steps, dist])
        for step in steps:
            if dist > 0:
                step_vec = last_pos + (vec / dist * step)
            else:
                step_vec = last_pos

            s10 = np.sqrt(np.power(step_vec[0][0], 2) + np.power(step_vec[1][0], 2))-dist_AP
            s20 = np.sqrt(np.power(XM2 - step_vec[0][0], 2) + np.power(step_vec[1][0], 2))-dist_AP
            alpha0 = np.arctan(-step_vec[1][0] / step_vec[0][0])
            beta0 = np.arctan(-step_vec[1][0] / (XM2 - step_vec[0][0]))
            gamma0 = 0
            max_length = np.sqrt(np.power(XM2,2)+np.power(H1,2))
            params0 = (s10, alpha0, gamma0)
            res = least_squares(kin, params0, args=(step_vec[0][0],step_vec[1][0],XM2,L1,L2,L3,0), bounds=([0,0,-np.pi/2],[max_length,np.pi,np.pi/2]))
            print(res.cost)
            """kin(res.x,step_vec[0][0],step_vec[1][0],XM2,L1,L2,L3,1)"""
            s1 = s10
            s2 = s20
            s1_plot = np.block([s1_plot,s1])
            s2_plot = np.block([s2_plot,s2])
            s10_plot = np.block([s10_plot, s10])
            s20_plot = np.block([s20_plot, s20])
            conv_line = command+' F'+str(feed)+' X'+str(s1)+' Y'+str(s2)+'\n'
            print(conv_line)
            f_new.write(conv_line)
        last_pos = np.array([[step_vec[0][0]], [step_vec[1][0]]])
    else:
        f_new.write(line)
f.close()
f_new.close()
plt.plot(s1_plot,s2_plot,'-k',s10_plot,s20_plot,'-b')
plt.show()
exit()

params0 = (0.5, 0.5, math.pi / 6, math.pi / 6, 0)
params0 = (1.19189973e+00, 3.39702998e-01, 1.68596951e-01, 5.97758461e-01, 3.95986675e-07)
res = minimize(kin, params0, args=(1.2, -0.25, 1.5, 0.05, 0.1, 0.05), method='Nelder-Mead', tol=1e-8)
print(res)
