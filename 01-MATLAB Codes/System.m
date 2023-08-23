clc;clear all;close all;
%% defining system parameters
m = 380;               % vehivle mass (kg)
lr = 0.6;              % distance between the centre of gravity and the rear axle(m)
r = 0.22;              % wheel radious (m)
Cr = 6000;             % Rear tire cornering stiffness (N/rad)
Ts = 0.01;             % sampling time (sec)
lf = 0.8;              % distance between the centre of gravity and the front axle (m)
dr = 0.82;             % tread at rear axle (m)
Cf = 6000;             % Rear tire cornering stiffness (N/rad)
Q = 0.0005*eye(4);     % Process noise covariance
R = 0.05*eye(4);       % Measurement noise covariance
Vx = 16.6;             % Velocity (m/sec)
l_pre = 1.5;
I = 136.08;            % Moment Inertia
%% defininge system matrices
A = [ (-2*(Cr+Cf)/(m*Vx))   (2*(Cr*lr - Cf*lf)/(m*Vx*Vx))-1  0  0
      (2*(Cr*lr - Cf*lf)/I) (-2*(Cr*lr^2 + Cf*lf^2)/(I*Vx))  0  0
      0                     1                                0  0
      Vx                    l_pre                            Vx 0];

B = [2*Cf/(m*Vx) 2*Cf*lf/(I) 0 0
     0           1/I         0 0]';

C = [0 1 0 0
     0 0 1 0
     0 0 0 1];

D = 0;
eig(A);
system = ss(A,B,C,D); % Continious System
discrete_System = c2d(system,Ts,'tustin');
eig(discrete_System.A)
