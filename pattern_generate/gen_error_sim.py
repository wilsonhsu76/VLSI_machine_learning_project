from scipy import signal
from scipy import misc
import numpy as np
import math
import random

#number of components(dic_index) to build filter
s = 3
kh = 3
kw = 3

# index number of dic
k = 100

# number of input channel
m = 64

# number of output channel
n = 1

# number of input image X
h = 14
w = 14

#define n-bit
bit_N = 16
rand_upperbound = math.pow(2, bit_N)

#define n-bit ( input: 0xFFFF<< Trun_bit_N)
Trun_bit_N = 8

#pic = np.array([[ 1,2,3],
#                [ 4,5,6],
#                [ 7,8,9]])
                  
#mask = np.array([[ 0,1,0],
#                 [ 0,2,0],
#                 [ 0,3,0]])

#rev_mask = np.flipud(np.fliplr(mask))

#grad = signal.convolve2d(pic, rev_mask, boundary='fill', mode='same')
#print(type(grad))
#print(grad)

#np.random.randint(5, size=(2, 4))

print('Truncate bit:')
print(Trun_bit_N)


# random produce D,X,C
D = np.random.randint(rand_upperbound, size=(k,m), dtype='int64')
X = np.random.randint(rand_upperbound, size=(m,h,w), dtype='int64')
C = np.random.randint(1,rand_upperbound, size=(n,s,kh,kw), dtype='int64')

# random fill I
I = np.zeros((n,s,kh,kw), dtype='int64')
for outCH in range(1,n+1):
    for row in range(1,kh+1):
        for col in range(1,kw+1):
            Rand_L = random.sample(range(1, k+1), s) #take s sample from k terms, no repeat
            for s_index in range(1,s+1):
                I[outCH-1, s_index-1, row-1, col-1] = Rand_L[s_index-1]

# W = use I,C to choose basis*C of dic
W = np.zeros((n,m,kh,kw), dtype='int64')
for outCH in range(1,n+1):
    for row in range(1,kh+1):
        for col in range(1,kw+1):
            tmp_mx = np.zeros((m,1,1), dtype='int64')
            L_C = C[outCH-1,:,row-1,col-1] # s terms
            L_I = I[outCH-1,:,row-1,col-1] # s terms
            for s_index in range(1,s+1):
                tmp = np.zeros((1,m), dtype='int64') #tmp:1*m
                tmp[0,:] = D[L_I[s_index-1]-1,:]*L_C[s_index-1] # take basis from dic and mult corrsponding C
                tmp_mx[:,0,0] += tmp[0,:] # sum up all s terms result into one (m,1,1) array
            W[outCH-1,:,row-1,col-1] = tmp_mx[:,0,0]          


# S = X*D
S = np.zeros((k,h,w), dtype='int64')
for dic_index in range(1,k+1):
    for in_cH in range(1, m+1):
        tmp_mx = np.zeros((m,h,w), dtype='int64')
        pic = X[in_cH-1, :, :]
        mask = D[dic_index-1, in_cH-1]
        tmp_mx[in_cH-1,:,:] = pic*mask
        S[dic_index-1,:,:] += tmp_mx[in_cH-1,:,:]

S_appro = np.zeros((k,h,w), dtype='int64')
for dic_index in range(1,k+1):
    for in_cH in range(1, m+1):
        tmp_mx = np.zeros((m,h,w), dtype='int64')
        pic = (X[in_cH-1, :, :]) & (0xFFFF << Trun_bit_N)
        mask = D[dic_index-1, in_cH-1] & (0xFFFF << Trun_bit_N)
        tmp_mx[in_cH-1,:,:] = pic*mask
        S_appro[dic_index-1,:,:] += tmp_mx[in_cH-1,:,:]  #S is truncated result

#print('S_appro VS S')
#print(np.max(S_appro))
#print(np.min(S_appro))
#print(np.max(S))
#print(np.min(S))

S_error_rates = np.zeros((k,h,w), dtype= float)
for dic_index in range(1,k+1):
    for row in range(1,h+1):
        for col in range(1,w+1):
            S_error_rates[dic_index-1,row-1,col-1] = abs(S[dic_index-1,row-1,col-1] - S_appro[dic_index-1,row-1,col-1]) / S[dic_index-1,row-1,col-1]

S_error_rate = np.mean(S_error_rates)
max_S_error_rate = np.max(S_error_rates)
print('S_error_rate:')
print(S_error_rate)
print('max_S_error_rate:')
print(max_S_error_rate)



# P : fill P by C and I information 
P = np.zeros((n,k,kh,kw), dtype='int64')
for outCH in range(1,n+1):
    for row in range(1,kh+1):
        for col in range(1,kw+1):
            L_C = C[outCH-1,:,row-1,col-1] # s terms
            L_I = I[outCH-1,:,row-1,col-1] # s terms
            for s_index in range(1,s+1):
                P[outCH-1,L_I[s_index-1]-1,row-1,col-1] = L_C[s_index-1]


# X*W: convolution between X and W
X_W = np.zeros((n,h,w), dtype='int64')
for outCH in range(1,n+1):
    tmp = np.zeros((h,w), dtype='int64')
    for in_CH in range(1,m+1):
        pic = X[in_CH-1,:,:]
        mask = W[outCH-1, in_CH-1, :, :]
        rev_mask = np.flipud(np.fliplr(mask)) #convoluion do unflipped mask 
        tmp = signal.convolve2d(pic, rev_mask, boundary='fill', mode='same')
        X_W[outCH-1,:,:] += tmp

# S*P: convolution between S and P
S_P = np.zeros((n,h,w), dtype='int64')
for outCH in range(1,n+1):
    tmp = np.zeros((h,w), dtype='int64')
    for k_index in range(1,k+1):
        pic = S[k_index-1,:,:]
        mask = P[outCH-1, k_index-1, :, :]
        rev_mask = np.flipud(np.fliplr(mask)) #convoluion do unflipped mask
        tmp = signal.convolve2d(pic, rev_mask, boundary='fill', mode='same')
        S_P[outCH-1,:,:] += tmp

# I/C/S in Figure2_Left
I_C_S_fig2Left = np.zeros((n,h,w), dtype='int64')
for outCH in range(1,n+1):
    tmp = np.zeros((h,w), dtype='int64')
    for s_index in range(1,s+1):
        for row in range(1,kh+1):
            for col in range(1,kw+1):
                pic = S[I[outCH-1,s_index-1,row-1,col-1]-1,:,:]
                coeff = C[outCH-1, s_index-1, row-1,col-1]
                tmp = pic*coeff
                #Below case only work for 3*3 filter case:
                # if other filter size should be genralized...
                if row==1 and col==1:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==1 and col==2:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                elif row==1 and col==3:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0
                elif row==2 and col==1:
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==2 and col==3:
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0
                elif row==3 and col==1:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==3 and col==2:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                elif row==3 and col==3:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0    
                I_C_S_fig2Left[outCH-1,:,:] += tmp


# I/C/S in Figure2_Left_appro
I_C_S_fig2Left_appro = np.zeros((n,h,w), dtype='int64')
for outCH in range(1,n+1):
    tmp = np.zeros((h,w), dtype='int64')
    for s_index in range(1,s+1):
        for row in range(1,kh+1):
            for col in range(1,kw+1):
                pic = (S_appro[I[outCH-1,s_index-1,row-1,col-1]-1,:,:]) #(S_approx)
                coeff = C[outCH-1, s_index-1, row-1,col-1] & (0xFFFF << Trun_bit_N) #(coeff originally is [15:0])
                tmp = pic*coeff
                #Below case only work for 3*3 filter case:
                # if other filter size should be genralized...
                if row==1 and col==1:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==1 and col==2:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                elif row==1 and col==3:
                    tmp = np.roll(tmp, 1, axis=0) #down
                    tmp[0,:] = 0
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0
                elif row==2 and col==1:
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==2 and col==3:
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0
                elif row==3 and col==1:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                    tmp = np.roll(tmp, 1, axis=1) #right
                    tmp[:,0] = 0
                elif row==3 and col==2:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                elif row==3 and col==3:
                    tmp = np.roll(tmp, -1, axis=0) #up
                    tmp[h-1,:] = 0
                    tmp = np.roll(tmp, -1, axis=1) #left
                    tmp[:,w-1] = 0    
                I_C_S_fig2Left_appro[outCH-1,:,:] += tmp

#print('I_C_S_fig2Left_appro VS I_C_S_fig2Left')
#print(np.max(I_C_S_fig2Left_appro))
#print(np.min(I_C_S_fig2Left_appro))
#print(np.max(I_C_S_fig2Left))
#print(np.min(I_C_S_fig2Left))

I_C_S_fig2Left_error_rates = np.zeros((n,h,w), dtype= float)
for outCH in range(1,n+1):
    for row in range(1,h+1):
        for col in range(1,w+1):
            I_C_S_fig2Left_error_rates[outCH-1,row-1,col-1] = abs(I_C_S_fig2Left[outCH-1,row-1,col-1] - I_C_S_fig2Left_appro[outCH-1,row-1,col-1]) / I_C_S_fig2Left[outCH-1,row-1,col-1]

#print(I_C_S_fig2Left_error_rates)
fig2Left_error_rate = np.mean(I_C_S_fig2Left_error_rates)
max_fig2Left_error_rate = np.max(I_C_S_fig2Left_error_rates)
print('\nfig2Left_error_rate:')
print(fig2Left_error_rate)
print('max_fig2Left_error_rate:')
print(max_fig2Left_error_rate)


# X*W_eq5: convolution between C and S (Take 27 convolutions.)
X_W_eq5 = np.zeros((n,h,w), dtype='int64')
for outCH in range(1,n+1):
    tmp = np.zeros((h,w), dtype='int64')
    for s_index in range(1,s+1):
        for row in range(1,kh+1):
            for col in range(1,kw+1):
                pic = S[I[outCH-1,s_index-1,row-1,col-1]-1,:,:]
                mask = np.zeros((kh,kw), dtype='int')
                #odd filter position
                mask[row-1, col-1] = C[outCH-1, s_index-1, row-1,col-1]
                rev_mask = np.flipud(np.fliplr(mask)) #convoluion do unflipped mask
                tmp = signal.convolve2d(pic, rev_mask, boundary='fill', mode='same')
                X_W_eq5[outCH-1,:,:] += tmp


print("\n\nX_W (max and min):")
print(np.max(X_W))
print(np.min(X_W))
print("S_P (max and min):")
print(np.max(S_P))
print(np.min(S_P))
print("X_W_eq5 (max and min):")
print(np.max(X_W_eq5))
print(np.min(X_W_eq5))
print("I_C_S_fig2Left (max and min):")
print(np.max(I_C_S_fig2Left))
print(np.min(I_C_S_fig2Left))

print("Does S_P and X_W have all the same elements?")
print((X_W == S_P).all())


print("Does X_W_eq5 and X_W have all the same elements?")
print((X_W == X_W_eq5).all())

print("Does I_C_S_fig2Left and X_W have all the same elements?")
print((X_W == I_C_S_fig2Left).all())

'''
file_abs = "test_gen.txt"
#D;X;W;S;P;C;I;X*W
with open(file_abs, "w") as f:
    #print D (k,m)
    for row in range(1,k+1):
        for col in range(1,m+1):
            f.write(str(D[row-1,col-1]))
            if (row!=k) or (col!= m): 
                f.write(",")
            else:
                f.write(",")
                f.write(";")

    #print X (m,h,w)
    for ch in range(1,m+1):
        for row in range(1,h+1):
            for col in range(1,w+1):
                f.write(str(X[ch-1,row-1,col-1]))
                if (ch!=m) or (row!=h) or (col!=w): 
                    f.write(",")
                else:
                    f.write(",")
                    f.write(";")

    #print W (n, m,kh,kw)
    for filter_n in range(1,n+1):
        for ch in range(1,m+1):
            for row in range(1,kh+1):
                for col in range(1,kw+1):
                    f.write(str(W[filter_n-1,ch-1,row-1,col-1]))
                    if (filter_n!=n) or (ch!=m) or (row!=kh) or (col!=kw): 
                        f.write(",")
                    else:
                        f.write(",")
                        f.write(";")

    #print S (k,h,w)
    for ch in range(1,k+1):
        for row in range(1,h+1):
            for col in range(1,w+1):
                f.write(str(S[ch-1,row-1,col-1]))
                if (ch!=k) or (row!=h) or (col!=w): 
                    f.write(",")
                else:
                    f.write(",")
                    f.write(";")

    #print P (n, k,kh,kw)
    for filter_n in range(1,n+1):
        for ch in range(1,k+1):
            for row in range(1,kh+1):
                for col in range(1,kw+1):
                    f.write(str(P[filter_n-1,ch-1,row-1,col-1]))
                    if (filter_n!=n) or (ch!=k) or (row!=kh) or (col!=kw): 
                        f.write(",")
                    else:
                        f.write(",")
                        f.write(";")

    #print C (n,s,kh,kw)
    for ch in range(1,n+1):
        for comb in range(1,s+1):
            for row in range(1,kh+1):
                for col in range(1,kw+1):
                    f.write(str(C[ch-1, comb-1,row-1, col-1]))
                    if (ch!=n) or (comb!=s) or (row!=kh) or (col!=kw): 
                        f.write(",")
                    else:
                        f.write(",")
                        f.write(";")

    #print I (n,s,kh,kw)
    for ch in range(1,n+1):
        for comb in range(1,s+1):
            for row in range(1,kh+1):
                for col in range(1,kw+1):
                    f.write(str(I[ch-1, comb-1,row-1, col-1]))
                    if (ch!=n) or (comb!=s) or (row!=kh) or (col!=kw): 
                        f.write(",")
                    else:
                        f.write(",")
                        f.write(";")

    #print X*W (n,h,w)
    for ch in range(1,n+1):
        for row in range(1,h+1):
            for col in range(1,w+1):
                f.write(str(X_W[ch-1,row-1,col-1]))
                if (ch!=n) or (row!=h) or (col!=w): 
                    f.write(",")
                else:
                    f.write(",")
                    f.write(";")

#    #print S*P (n,h,w)
#    for ch in range(1,n+1):
#        for row in range(1,h+1):
#            for col in range(1,w+1):
#                f.write(str(S_P[ch-1,row-1,col-1]))
#                if (ch!=n) or (row!=h) or (col!=w): 
#                    f.write(",")
#                else:
#                    f.write(";")
'''



    
