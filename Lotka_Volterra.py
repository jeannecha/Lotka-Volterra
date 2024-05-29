#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle de Lotka-Volterra
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 5)

# Résolution du problème avec RK4 :
def RK4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = len(t)
    nx = len(x0)
    x = np.zeros((nx,nt))
    x[:,0] = x0
    
    for k in range(nt-1):
        k1 = dt*f(t[k], x[:,k]);
        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2)
        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2)
        k4 = dt*f(t[k] + dt, x[:,k] + k3)
        
        dx=(k1 + 2*k2 + 2*k3 +k4)/6
        x[:,k+1] = x[:,k] + dx;  
    
    return x, t


# Modèle basique Lotka-Volterra :
def dynamique(x, a, b, c, d):
    xp = np.array([a*x[0] - b*x[0]*x[1], d*x[0]*x[1] - c*x[1]])
    return xp

# choix des paramètres des taux de reproduction, mortalité... :
a, b, c, d = 0.1, 0.01, 0.05, 0.001
f= lambda t, x : dynamique(x, a, b, c, d)

# intervalle de temps pour faire la résolution :
t0 = 0
tf = 350
dt = 0.01

# conditions initiales choisies arbitrairement ([nombre de proies, nombre de prédateurs]) :
X0 = [np.array([20, 5]), np.array([50, 15]), np.array([5, 20]), np.array([15, 50])] 

# Tracé dans l'espace des phases :
plt.figure()
for x0 in X0:
    x, t = RK4(f, x0, t0, tf, dt)
    plt.plot(x[0,:], x[1,:], label=""+str(x0[0])+" - "+str(x0[1]))

plt.xlabel("Proies")
plt.ylabel("Prédateurs")
plt.grid()
plt.legend(title="Conditions initiales :\n proies - prédateurs")
plt.show()

# Évolution des individus au cours du temps : 
for x0 in X0:
    x, t = RK4(f, x0, t0, tf, dt)
    plt.figure()
    plt.plot(t, x[0,:], "r", label="Proies")
    plt.plot(t, x[1,:], "b", label="Prédateurs")
    plt.xlabel("Temps (arbitraire)")
    plt.ylabel("Nombre d'individus")
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    
#%%

# Variante du modèle avec capacité de charge pour limiter le nombre de proies :
def variante(x, a, b, c, d, K):
    xp = np.array([x[0]*(a-a/K*x[0]-b*x[1]), x[1]*(-c+d*x[0])])
    return xp

# Tracé de la trajectoire suivie dans l'espace des phases :
def trace_trajectoire(Xsolution):
    plt.plot(Xsolution[0,:], Xsolution[1,:], color= 'tab:orange') # tracé
    plt.plot([Xsolution[0,0]], [Xsolution[1,0]], 'sk', label="Condition initiale") # début

# intervalle de temps pour faire la résolution :
t0 = 0
tf = 600
dt = 0.01

# valeurs des paramètres et condition initiale :
a, b, c, d, K = 0.1, 0.01, 0.05, 0.001, 200
fvariante= lambda t, x : variante(x, a, b, c, d, K)
x0 = np.array([50, 15])

# affichage des vecteurs :
xmin=20
xmax=75
ymin=4
ymax=16
Nx= 15  # résolution du quadrillage en x
Ny= 15  # résolution du quadrillage en y

xrange = np.linspace(xmin, xmax, Nx)
yrange = np.linspace(ymin, ymax, Ny)
x_grille, y_grille = np.meshgrid(xrange, yrange)
m,n = np.shape(x_grille)
xpoint_grille = np.zeros((m,n))
ypoint_grille = np.zeros((m,n))

NI, NJ = x_grille.shape
for i in range(NI):
    for j in range(NJ):
        xg = x_grille[i, j]
        yg = y_grille[i, j]
        Xpoint = variante([xg, yg], a, b, c, d, K)
        xpoint_grille[i,j] = Xpoint[0]
        ypoint_grille[i,j] = Xpoint[1]


# Calcul de la solution :
x, t = RK4(fvariante, x0, t0, tf, dt)

# Tracé dans l'espace des phases :
plt.figure()
plt.quiver(x_grille, y_grille, xpoint_grille, ypoint_grille, color='dimgrey', width=0.003, scale=30)
trace_trajectoire(x)
plt.xlabel("Proies")
plt.ylabel("Prédateurs")
plt.grid()
plt.legend()
plt.show()

# Évolution des espèces au cours du temps :
plt.figure()
plt.plot(t, x[0,:], "r", label="Proies")
plt.plot(t, x[1,:], "b", label="Prédateurs")
plt.xlabel("Temps (arbitraire)")
plt.ylabel("Nombre d'individus")
plt.grid()
plt.legend(loc='best')
plt.show()


#%%

# modèle de chaine alimentaire omnivore :
def modele_omnivore(X, d1, d2, alpha, beta, gamma, gamma_b, delta):
    x, y, z = X
    xp = x*(1-x-y-gamma_b*z)
    yp = y*(-d1 +alpha*x - beta*z)
    zp = z*(-d2 + gamma*x +delta*y)
    return np.array([xp, yp, zp])

# modèle de chaine alimentaire complète :
def chaine_alim(X, d1, d2, alpha, beta, gamma, gamma_b, delta):
    x, y, z = X
    yp = y*(-d1 +alpha*x - beta*z)
    zp = z*(-d2 + gamma*x +delta*y)
    xp = x*(1-x-y-gamma_b*z) - d1*yp - d2*zp
    return np.array([xp, yp, zp])

t0 = 0
tf = 70
dt = 0.01

# deux sets de paramètres, le premier pour l'extinction d'une espèce à temps long, l'autre pour la coexistence :
d1, d2, alpha, beta, gamma, gamma_b, delta = 0.4, 0.24, 2.5, 0.2, 0.25, 1, 0.25
#d1, d2, alpha, beta, gamma, gamma_b, delta = 0.8, 0.24, 2, 0.2, 0.25, 1, 0.6

f1 = lambda t, x : modele_omnivore(x, d1, d2, alpha, beta, gamma, gamma_b, delta)
f2 = lambda t, x : chaine_alim(x, d1, d2, alpha, beta, gamma, gamma_b, delta)
X0 = [np.array([0.3, 0.2, 0.1]), np.array([0.4, 0.2, 0.5])]

# Différents affichages de figures :
for x0 in X0 :
    x1, t1 = RK4(f1, x0, t0, tf, dt)
    plt.figure()
    plt.plot(t1, x1[0,:], "r", label="Ressources")
    plt.plot(t1, x1[1,:], "b", label="Proies")
    plt.plot(t1, x1[2,:], "g", label="Prédateurs")
    plt.xlabel("Temps (arbitraire)")
    plt.ylabel("Densité d'individus")
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    
      
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x0[0], x0[1], x0[2], 'sk', label="Point initial")
    ax.plot(x1[0, :], x1[1, :], x1[2, :])
    ax.plot(x1[0, -1], x1[1, -1], x1[2, -1], 'sr', label='Point final')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(25, 70)
    ax.legend()
    
    x2, t2 = RK4(f2, x0, t0, tf, dt)
    plt.figure()
    plt.plot(t2, x2[0,:], "r", label="Ressources")
    plt.plot(t2, x2[1,:], "b", label="Proies")
    plt.plot(t2, x2[2,:], "g", label="Prédateurs")
    plt.xlabel("Temps (arbitraire)")
    plt.ylabel("Densité d'individus")
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    
      
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x0[0], x0[1], x0[2], 'sk', label="Point initial")
    ax.plot(x2[0, :], x2[1, :], x2[2, :])
    ax.plot(x2[0, -1], x2[1, -1], x2[2, -1], 'sr', label='Point final')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(25, 70)
    ax.legend()
