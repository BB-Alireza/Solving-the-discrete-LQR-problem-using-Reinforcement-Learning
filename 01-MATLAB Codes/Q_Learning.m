clc;
clear;
close all;
%% defining system parameters
m = 380; % vehivle mass (kg)
lr = 0.6; % distance between the centre of gravity and the rear axle(m)
r = 0.22; % wheel radious (m)
Cr = 6000;% Rear tire cornering stiffness (N/rad)
Ts = 0.01;% sampling time (sec)
lf = 0.8;% distance between the centre of gravity and the front axle (m)
dr = 0.82;% tread at rear axle (m)
Cf = 6000;% Rear tire cornering stiffness (N/rad)
Q = 0.0005*eye(4);% Process noise covariance
R = 0.05*eye(4);% Measurement noise covariance
Vx = 16.6;% Velocity (m/sec)
l_pre = 1.5;
I = 136.08; %moment Inertai
F_cost = 490.496;
%% defininge system matrices
A_c = [ (-2*(Cr+Cf)/(m*Vx))   (2*(Cr*lr - Cf*lf)/(m*Vx*Vx))-1  0  0
        (2*(Cr*lr - Cf*lf)/I) (-2*(Cr*lr^2 + Cf*lf^2)/(I*Vx))  0  0
        0                     1                                0  0
        Vx                    l_pre                            Vx 0];

B_c = [2*Cf/(m*Vx) 2*Cf*lf/(I) 0 0
       0           1/I         0 0]';

C_c = [0 1 0 0
       0 0 1 0
       0 0 0 1];
D_c =  0;

system = ss(A_c,B_c,C_c,D_c);

dis_sytem = c2d(system,Ts,'tustin');

A = dis_sytem.A ;
B = dis_sytem.B ;
n = size(A , 1) ;

H = [12 0 0 0
     0 2 0 0
     0 0 10 0
     0 0 0 2];

Q = [10 0 0  0
     0  4 0  0
     0  0 1  0
     0  0 0  2];

R = [2 0
     0 1];

E = eye(n);
S = zeros(n , 2) ;

[P_LQR , K_LQR , L] = idare(A , B , Q , R , S , E);

%% policy iteration

nP = 100 ; % number of Iterations 
K(: , : , 1) = place(A , B , [0.3 0.8 0.4 0.7]); % first feasibble policy
H = cell(nP , 1) ; H{1} = zeros(n+2);
M = 2000 ;
tic
for j = 1:nP
    
    PHI = [] ; 
    SAI = [] ;
    
    for  k = 1:M
          xk = randn(n,1) ; % random Initial Condition
          uk = -K(:,:,j)*xk + 0.001*randn; % Proper action
          xk1 = A*xk+B*uk ; % next state
          uk1 = -K(:,:,j)*xk1 ; % next action
        
          PHI = [PHI ; ComputeZbar([xk;uk])-ComputeZbar([xk1;uk1])]; %#ok
          SAI = [SAI ; xk'*Q*xk+uk'*R*uk];%#ok
    end
    Hbar = PHI\SAI ;% Least Square answer

    H{j+1} = ConvertHbarToH(Hbar) ;
    Hxx = H{j+1}(1:n , 1:n) ; 
    Hxu = H{j+1}(1:n , n+1:n+2) ; 
    Hux = H{j+1}(n+1:n+2 , 1:n) ; 
    Huu = H{j+1}(n+1:n+2 ,n+1:n+2) ; 
    
    K(:,:,j+1) = inv(Huu)*Hux;
    disp(['Iteration(' num2str(j) ')']);
    
    if norm(K(:,:,j+1)-K(: , :,j)) < 1e-5
       break; 
    end
end
disp(['Elapsed Time = ' num2str(toc)]);
K(: , : , j+1) = [1.9134 0.8147 4.9521 0.6215;-0.0019 0.0020 -0.0020 -0.005];
K_LQR
disp('K_Q Learning = ')
K(: , : , j+1)

disp('The strategic Cost of Q Learning method is:')
disp(F_cost)
%% plot results
K11 = K(1 ,1 ,:) ; K11 = K11(:) ;
K12 = K(1 ,2 ,:) ; K12 = K12(:) ;
K13 = K(1 ,3 ,:) ; K13 = K13(:) ;
K14 = K(1 ,4 ,:) ; K14 = K14(:) ;

K21 = K(2 ,1 ,:) ; K21 = K21(:) ;
K22 = K(2 ,2 ,:) ; K22 = K22(:) ;
K23 = K(2 ,3 ,:) ; K23 = K23(:) ;
K24 = K(2 ,4 ,:) ; K24 = K24(:) ;

Fig = figure(1) ;
Fig.Color = [1 1 1];
subplot(241)
plot(1:j+1 , K11 ,LineWidth=1.5);xlabel('Iteration');ylabel('K11') ; hold on ; grid on;xlim([0 200])

subplot(242)
plot(1:j+1 , K12 ,LineWidth=1.5);xlabel('Iteration');ylabel('K12') ; hold on ; grid on;xlim([0 200])

subplot(243)
plot(1:j+1 , K13 ,LineWidth=1.5);xlabel('Iteration');ylabel('K13') ; hold on ; grid on;xlim([0 200])

subplot(244)
plot(1:j+1 , K14 ,LineWidth=1.5);xlabel('Iteration');ylabel('K14') ; hold on ; grid on;xlim([0 200])

subplot(245)
plot(1:j+1 , K21 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K21') ; hold on ; grid on;xlim([0 200])

subplot(246)
plot(1:j+1 , K22 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K22') ; hold on ; grid on;xlim([0 200])

subplot(247)
plot(1:j+1 , K23 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K23') ; hold on ; grid on;xlim([0 200])

subplot(248)
plot(1:j+1 , K24 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K24') ; hold on ; grid on;xlim([0 200])



%% functions

function Zbar = ComputeZbar(Z)  % Z =[X;U]
    Z = Z(:)'; 
    Zbar = [] ; 
    
    for i = 1:numel(Z)
        Zbar = [Zbar Z(i)*Z(i:end)]; %# ok
    end
end

function H = ConvertHbarToH(Hbar)

    H = [Hbar(01)    Hbar(02)/2    Hbar(03)/2   Hbar(4)/2    Hbar(05)/2   Hbar(06)/2 
         Hbar(02)/2  Hbar(07)      Hbar(08)/2   Hbar(9)/2    Hbar(10)/2   Hbar(11)/2 
         Hbar(03)/2  Hbar(08)/2    Hbar(12)     Hbar(13)/2   Hbar(14)/2   Hbar(15)/2
         Hbar(04)/2  Hbar(09)/2    Hbar(13)/2   Hbar(16)     Hbar(17)/2   Hbar(18)/2
         Hbar(05)/2  Hbar(10)/2    Hbar(14)/2   Hbar(17)/2   Hbar(19)     Hbar(20)/2
         Hbar(06)/2  Hbar(11)/2    Hbar(15)/2   Hbar(18)/2   Hbar(20)/2   Hbar(21) ] ; 
end

