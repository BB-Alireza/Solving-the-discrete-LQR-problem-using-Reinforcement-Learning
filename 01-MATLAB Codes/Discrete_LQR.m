clc;clear all;close all;

%% defining system parameters
m = 380;               % vehivle mass (kg)
lr = 0.6;              % distance between the centre of gravity and the rear axle(m)
r = 0.22;              % wheel radious (m)
Cr = 6000;             % Rear tire cornering stiffness (N/rad)
T = 0.01;             % sampling time (sec)
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
D_c = 0;

Cont_System = ss(A_c,B_c,C_c,D_c);
Dis_sytem = c2d(Cont_System,T);
%% discrete LQR
A = Dis_sytem.A;
B = Dis_sytem.B;

H = [12 0 0 0
     0 2 0 0
     0 0 10 0
     0 0 0 2];

Q = [2.5  0   0   0
     0    0.5 0   0
     0    0   2.5 0
     0    0   0   0.5];

R = [2 0
     0 1];

N = 400;% Number of samples
F = zeros(2 , 4 , N);
P = zeros(4 , 4 , N);
P(:,:,1) = H; % initial condition for difference equation of P
tic
for k = 2:N-1

    F(:,:,N-k) = -(R + B'*P(:,:,k-1)*B)^(-1)*B'*P(:,:,k-1)*A ;
    P(:,:,k) = (A + B*F(:,:,N-k))'*P(:,:,k-1)*(A + B*F(:,:,N-k))+F(:,:,N-k)'*R*F(:,:,N-k)+Q ;

end
disp(['Elapsed Time = ' num2str(toc)]);


%%% F = [F11 F12 F13 F14 
%        F21 F22 F23 24]

F11 = F(1 ,1 ,:) ; F11 = F11(:) ;
F12 = F(1 ,2 ,:) ; F12 = F12(:) ;
F13 = F(1 ,3 ,:) ; F13 = F13(:) ;
F14 = F(1 ,4 ,:) ; F14 = F14(:) ;

F21 = F(2 ,1 ,:) ; F21 = F21(:) ;
F22 = F(2 ,2 ,:) ; F22 = F22(:) ;
F23 = F(2 ,3 ,:) ; F23 = F23(:) ;
F24 = F(2 ,4 ,:) ; F24 = F24(:) ;

Fig1 = figure(1);
Fig1.Color = [0.9 0.9 0.9];
subplot(2,1,1)
plot(1:N ,F11 , 1:N , F12 , 1:N ,F13 , 1:N ,F14 , 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('F1(N-k)');
legend('F11' , 'F12' , 'F13' , 'F14')

subplot(2,1,2)
plot(1:N ,F21 , 1:N , F22 , 1:N ,F23 , 1:N ,F24 , 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('F2(N-k)');
legend('F21' , 'F22' , 'F23' , 'F24')


%%%% testing the simulation
K_dlqr = dlqr(A,B,Q,R,zeros(4,2));
disp('gains obtained from dlqr formula:')
disp(-K_dlqr)

disp('Final Gains Using Recursive Formula:')
disp(F(:,:,1))

%% plot the states injecting the optimal input (u_Star)
x = zeros(4 , N) ; 
x(: , 1) = [1;2;.2;-4]; % Initial Condition
u = zeros(2 , 1 , N) ;

for i = 1:N-1
    u(:,:,i) = F( : , : , i)*x(: ,i);
    x(: , i+1) = A*x(: , i) + B*u(:,:,i); 
end

u1 = u(1,:,:);
u1 = u1(:);

u2 = u(2,:,:);
u2 = u2(:);

Fig2 = figure(2);
Fig2.Color = [0.9 0.9 0.9];
subplot(311)
plot(1:N ,x(1,:),1:N,x(2,:),1:N,x(3,:),1:N,x(4,:),LineWidth=1.5)
grid on
xlabel('Iteration');
ylabel('System States');
legend('X1' , 'X2' , 'X3' , 'X4')

subplot(312)
plot(1:N ,u1,'r',LineWidth=1.5)
grid on
xlabel('Iteration');
ylabel('Control Input 1');
legend('U1')

subplot(313)
plot(1:N,u2,LineWidth=1.5)
grid on
xlabel('Iteration');
ylabel('Control Input 2');
legend('U2')

%%% total optimal cost
disp('----*----*----*')
disp('Total Optimal cost using formula:')
J_opt = 0.5 * x(:,1)' * P(:,:,N-1) * x(:,1  );
disp(J_opt)

%% calculate  total cost using the cost function formula 
cost = 0 ; 
for o = 1:N-1
     cost = cost + 0.5 * (x(:,o)' * Q * x(:,o) + u(:,o)' * R * u(:,o));
end
disp('Total Optimal cost obtained from cost function summation:')
cost + 0.5 * x(:,N)' * H * x(:,N);
disp(cost)