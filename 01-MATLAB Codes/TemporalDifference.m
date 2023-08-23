clc;
clear;
close all;
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

Q = [10 0 0 0
     0 2 0 0
     0 0 10 0
     0 0 0 2];

R = [2 0
     0 1];

E = eye(n);
S = zeros(n , 2) ;

[P_LQR , K_LQR , L] = idare(A , B , Q , R , S , E);

%% Value Iteration
nP = 200 ; %number of iterations
K = zeros(2 , 4 , nP);
K(: , : , 1) = place(A , B , [0.3 0.8 0.4 0.7]); % first feasibble policy
P = cell(nP , 1) ; P{1} = zeros(n);
M = 10;
x = zeros(4,nP);
x(:,1) = [1;-0.2;.2;-4];
tic
for j = 1:nP
    
    PHI = [] ; 
    SAI = [] ;
    
    for  k = 1:M
         xk = randn(n,1) ;
         uk = -K(:,:,j)*xk + 0.01*randn;
         xk1 = A*xk + B*uk ; 
        
         PHI = [PHI ; ComputeXbar(xk)-ComputeXbar(xk1)]; %#ok
         SAI = [SAI ; xk'*Q*xk+uk'*R*uk];%#ok
    end
    Pbar = PHI\SAI ; % Least Square answers
    
    P{j+1} = ConvertPbarToP(Pbar);

    K(: , : , j+1) = inv((R + B' * P{j+1} * B)) * (B' * P{j+1} * A);
    
    x(:,j+1) = A*x(:,j) - B * (K(:,:,j) * x(:,j));

    disp(['Iteration(' num2str(j) ')']);
    
    if norm(K(: , : , j+1) - K(: , : , j)) < 1e-5
       break; 
    end
end
disp(['Elapsed Time = ' num2str(toc)]);

K_LQR
disp('K_TD = ')
K_final = K(: , : , j+1);
disp(K_final)

P_LQR
disp('P_TD = ')
P_final = P{j+1};
disp(P_final)

disp('The strategic Cost of Temporal Difference method is:')
disp(0.5 * x(:,1)' * P{j+1} * x(:,1))

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
Fig.Color = [0.9 0.9 0.9];
subplot(241)
plot(1:j+1 , K11 ,LineWidth=1.5);xlabel('Iteration');ylabel('K11') ; hold on ; grid on;xlim([0 j])

subplot(242)
plot(1:j+1 , K12 ,LineWidth=1.5);xlabel('Iteration');ylabel('K12') ; hold on ; grid on;xlim([0 j])

subplot(243)
plot(1:j+1 , K13 ,LineWidth=1.5);xlabel('Iteration');ylabel('K13') ; hold on ; grid on;xlim([0 j])

subplot(244)
plot(1:j+1 , K14 ,LineWidth=1.5);xlabel('Iteration');ylabel('K14') ; hold on ; grid on;xlim([0 j])

subplot(245)
plot(1:j+1 , K21 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K21') ; hold on ; grid on;xlim([0 j])

subplot(246)
plot(1:j+1 , K22 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K22') ; hold on ; grid on;xlim([0 j])

subplot(247)
plot(1:j+1 , K23 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K23') ; hold on ; grid on;xlim([0 j])

subplot(248)
plot(1:j+1 , K24 ,'r',LineWidth=1.5);xlabel('Iteration');ylabel('K24') ; hold on ; grid on;xlim([0 j])


Fig2 = figure(2);
Fig2.Color = [0.9 0.9 0.9];
plot(1:nP,x(1,1:nP),LineWidth=1.8),title('States')
hold on
plot(1:nP,x(2,1:nP),LineWidth=1.8)
hold on
plot(1:nP,x(3,1:nP),LineWidth=1.8)
hold on
plot(1:nP,x(4,1:nP),LineWidth=1.8)
hold on
grid on
xlabel('Iteration')
ylabel('States')
legend('X1','X2','X3','X4')



%% functions

function Xbar = ComputeXbar(X)
    X = X(:)'; 
    Xbar = [] ; 
    
    for i = 1:numel(X)
        Xbar = [Xbar X(i)*X(i:end)]; %# ok
    end
end

function P = ConvertPbarToP(Pbar)

    P = [Pbar(1)    Pbar(2)/2    Pbar(3)/2  Pbar(4)/2
         Pbar(2)/2  Pbar(5)      Pbar(6)/2  Pbar(7)/2
         Pbar(3)/2  Pbar(6)/2    Pbar(8)    Pbar(9)/2
         Pbar(4)/2  Pbar(7)/2    Pbar(9)/2  Pbar(10)] ; 
end


