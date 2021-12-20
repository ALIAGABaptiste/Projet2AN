# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 07:41:58 2021

@author: baptj
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import random 

#Question 1 et QUestion 2

def CubicInterp(t, y, k ):
    
    N = 4
    A = np.vander(t, N, increasing=True)    
    Coef = npl.solve(A,y)
    
    P = Coef[0]+  Coef[1]*k + Coef[2]*k**2 + Coef[3]*k**3
    Pderiv = Coef[1] + 2*Coef[2]*k + 3*Coef[3]*k**2 #QUESTION 2
    return (P,Pderiv) #Dans la question 1 on ne retourne que P

print('Les interpolations obtenues pour le question 1 et 2 sont',CubicInterp(np.array([1,2,3,4]),np.array([1,2,3,4]), np.pi))
print('')

#%%

#Question 3

f =np.loadtxt('data_f.txt')
g =np.loadtxt('data_g.txt')  #On introduit les données du problème
t =np.loadtxt('data_t.txt')  

def InterpFunctionQ3(t,y,k):
    i=0
    while t[i] < k:
        i = i+1
        T4val = np.array([t[i-2],t[i-1],t[i],t[i+1]])
        Y4val = np.array([y[i-2],y[i-1],y[i],y[i+1]])
        P = (CubicInterp(T4val, Y4val, k))[0]
        Pd = (CubicInterp(T4val,Y4val,k))[1]
    return (P, Pd)

k = 0.003
print ('La valeur de f interpolée à 3 ms est',InterpFunctionQ3(t,f,k)[0])
print('')
print ('La valeur de g interpolée à 3 ms est',InterpFunctionQ3(t,g,k)[0])
print('')

#%%

#Question 4

def InterpFunctionQ4(t,y,k):
    P = np.zeros(len(k))
    Pd = np.zeros(len(k))
    for j in range(len(k)):
        i = 0
        while t[i] < k[j] :
            i = i+1
        T4val = np.array([t[i-2],t[i-1],t[i],t[i+1]])
        Y4val = np.array([y[i-2],y[i-1],y[i],y[i+1]])
        P[j] = (CubicInterp(T4val, Y4val, k[j]))[0]
        Pd[j] = (CubicInterp(T4val, Y4val, k[j]))[1]
    return (P, Pd)

k = np.linspace(0.001,0.003,100) 
  
#Affichage des courbes d'interpolations

plt.title('Courbes d interpolations des signaux f(t) et g(t) pour une intervalle de 1 à 3 ms')
plt.plot(k,InterpFunctionQ4(t, f, k)[0],'y')
plt.plot(k,InterpFunctionQ4(t, g, k)[0],'b')
plt.ylabel('tension en volt (V)')
plt.xlabel('temps en secondes (s)')
plt.legend(('f(t)','g(t)'))
plt.figure()
plt.title('Déivées des courbes d interpolations des signaux f(t) et g(t) pour une intervalle de 1 à 3 ms')
plt.plot(k,InterpFunctionQ4(t, f, k)[1],'y')
plt.plot(k,InterpFunctionQ4(t, g, k)[1],'b')
plt.ylabel('tension en volt (V)')
plt.xlabel('temps en secondes (s)')
plt.legend(('df(t)','dg(t)'))

#%%

#Question 5

# =============================================================================
# Après une analyse du graphique donné, la valeur Phi0 du déphasage grossièrement mesurée est de 0.6 ms
# =============================================================================

phi0 = 0.0006

#%%

#Question 6

T = k #L'intervalle d'étude étant [0.001,0.003], cela correspond aux valeurs de k
F = InterpFunctionQ4(t,f,T)[0]
G = InterpFunctionQ4(t,g,T)[0]

G0 = InterpFunctionQ4(t,g,T+phi0)[0]
DG0 = InterpFunctionQ4(t,g,T+phi0)[1] 

plt.title('Courbes de comparaison des interpolations de f(t), g(t) et g(t+phi0)')
plt.plot(T,F,'b')
plt.plot(T,G,'g')
plt.plot(T,G0,'y')
plt.xlabel('temps en secondes (s)')
plt.ylabel('tension en Volt (V)')
plt.legend(('f(t)','g(t)','g(t+phi0)'))

#%%

#Question 7

Frésidu = F-G0
Frésidu = InterpFunctionQ4(t, F-G0, T)[0]
plt.figure()
plt.title('Courbe du résidu f(t) - g(t+phi0)')
plt.plot(k,Frésidu)
plt.xlabel('temps en secondes (s)')
plt.ylabel('tension en Volt (V)')
plt.legend(('résidu',''))

J = np.dot(np.transpose(Frésidu),Frésidu) #On peut aussi faire J = (npl.norm(Frésidu))**2 

J0 = 0
for i in range(6,17): #On prend les valeurs 6 à 17 car se sont les valeurs situées dans l'intervalles I dans les données
    J0 = J0 + (f[i]-g[i])**2
 
print('')
print('J(phi0) =',J)
print('')
print('J(0) =',J0)

#On peut représenter la courbe du résidu avec les courbes de f et g pour voir un ordre de grandeur

#On peut conclure que notre estimation initiale est bonne mais qu'elle n'est pas parfaite car J(phi0) n'est pas égal à 0

#%%

#Question 8 et 9 faites sur papier => Voir  le rapport

#%%

#Question 10

i = 0
phi = phi0 
DG =  DG0
Gphi = G0
dphi = 1 #Cette valeur est choisie arbitrairement, elle va se modifier dans la boucle

while np.abs(dphi) > 10e-16 or i < 100: 
#On arrete l’algorithme soit quand le nombre d’iterations
#est superieur a 100 (i), soit quand la correction de phi sera suffisament petite.

    i = i+1
    DGT = np.transpose(DG)
    dphi = (np.dot(DGT,(F-Gphi)))/(np.dot(DGT,DG))
    phi = phi + dphi #On corrige dphi
    Gphi = InterpFunctionQ4(t,g,k+phi)[0]
    DG = InterpFunctionQ4(t,g,k+phi)[1]
 
print('')
print ('On obtient au final un déphasage en f et g de',phi,'secondes')

#%%

#Question 11

#Courbe d'interpolation de f(t) et g(t+phi) avec le nouveau résidu obtenu
Frésidu = F-G0
Frésidu = InterpFunctionQ4(t, F-G0, k)[0] #On recalcule le résidu

plt.figure()
plt.title('Courbes des signaux f(t) et g(t+phi)')
plt.plot(k,F,'g')
plt.plot(k,Gphi,'y')
plt.plot(k,Frésidu,'b')
plt.xlabel('temps en secondes (s)')
plt.ylabel('tension en volt (V)')
plt.legend(('f(t)','g(t+phi)','résidu')) #On met le résidu avec les courbes comme mentionné dans la question 7

#On observe sur la figure que f(t) et g(t+phi) sont confondues

#Préparation des courbes de convergence

#On prépare les données classques
i = 0
phi = phi0
DG = DG0
Gphi = G0
dphi = 1

#On introduit le résidu, le nombre d'itérations et la norme de la correction

résidu = []
itera = []
correct = []

#On effectue la même boucle qu'à la question 10

while np.abs(dphi) > 10e-16 or i < 100:
    
    résidu.append(npl.norm(F-Gphi))
    itera.append(i)
    i = i+1
    DGT = np.transpose(DG)
    dphi = (np.dot(DGT,(F-Gphi)))/(np.dot(DGT,DG))
    correct.append(np.abs(dphi))
    phi = phi + dphi
    Gphi = InterpFunctionQ4(t,g,k+phi)[0]
    DG = InterpFunctionQ4(t,g,k+phi)[1]


résidu = np.array(résidu)
itera = np.array(itera)
correct = np.array(correct)

#On trace les deux courbes de convergence 

plt.figure()
plt.title('Convergence de la mesure du résidu')
plt.semilogy(itera,résidu,'r') #Echelle log
plt.xlabel('nombre d itérations')
plt.ylabel('tension en Volt (V)')
plt.legend(('résidu',''))
plt.figure()
plt.title('Convergence de la norme de la correction')
plt.semilogy(itera,correct,'y') #Echelle log
plt.xlabel('nombre d itérations')
plt.ylabel('tension en Volt (V)')
plt.legend(('|phi|',''))

#On a relancé le programme avec différente initialisation => Voir rapport

vf = (0.04)/phi
print('')
print ('Vitesse initiale :',vf,'m/s.')

#%%

#Question 12

#Bruitage des signaux initiaux

bruitage = 0.2 #On ajoute un bruitage choisit totalement aléatoirement 
fb = np.zeros(len(f)) #On applique le bruitage sur f
for i in range(len(f)):
    fb[i] = f[i] + random.uniform(- bruitage,bruitage)
    
plt.figure()
plt.title('Courbe du signal de f bruité')
plt.plot(k,InterpFunctionQ4(t, f, k)[0],'y')
plt.plot(k,InterpFunctionQ4(t, fb, k)[0],'b')
plt.xlabel('temps en secondes(s)')
plt.ylabel('tension en Volt (V)')
plt.legend(('f(t)','f(t) bruité avec une valeur de 0.2'))

#Etude de la sensibilité du résultat au bruit capteur

def SensibiliteBruit (b):

    f = np.loadtxt("data_g.txt") #Il faut réintroduire f pour éviter une erreur
    bruitage = b
    fb = f
    for i in range(len(f)):
        fb[i] = f[i] + random.uniform(- bruitage,bruitage)
    f = fb

    phi0 = 0.0006

    F = InterpFunctionQ4(t,f,k)[0]
    G0 = InterpFunctionQ4(t,g,k + phi0)[0]
    DG0 = InterpFunctionQ4(t,g,k + phi0)[1]

    i = 0
    phi = phi0
    DG = DG0
    Gphi = G0
    dphi = 1

    while np.abs(dphi) > 10e-16 or i < 100:

        i = i+1
        DGT = np.transpose(DG)
        dphi = (np.dot(DGT,(F-Gphi)))/(np.dot(DGT,DG))
        phi = phi + dphi
        Gphi = InterpFunctionQ4(t,g,k+phi)[0]
        DG = InterpFunctionQ4(t,g,k+phi)[1]
    
    V = (0.04)/phi
    ErV = (np.abs(vf-V))/vf
    return ErV, V

bruit = np.linspace(0,1,10) #Vecteurs choisi arbitrairement
erreur = []    
for i in range(len(bruit)):
    erreur.append(SensibiliteBruit(bruit[i]*4)[0])

erreur = np.array(erreur)

plt.figure()
plt.title('Erreur de la vitesse en fonction du pourcentage de bruit')
plt.plot(bruit*100,erreur)
plt.xlabel('Bruit en %')
plt.ylabel('Erreur sur la vitesse')





