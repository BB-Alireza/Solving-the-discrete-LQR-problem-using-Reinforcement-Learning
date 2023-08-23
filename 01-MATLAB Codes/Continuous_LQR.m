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
A = [ (-2 * (Cr+Cf)/(m*Vx))    (2*(Cr*lr - Cf*lf)/(m*Vx*Vx))-1  0  0
      (2*(Cr*lr - Cf*lf)/I)    (-2*(Cr*lr^2 + Cf*lf^2)/(I*Vx))  0  0
      0                        1                                0  0
      Vx                       l_pre                            Vx 0];

B = [2*Cf/(m*Vx) 2*Cf*lf/(I) 0 0
     0           1/I         0 0]';

C = [0 1 0 0
     0 0 1 0
     0 0 0 1];
D = 0;

system = ss(A,B,C,D);

eig(A);
%% Finding optimal solution with HJB (Continuous LQR)
t0 = 0;
tf = 15;
Ts = 0.01;
t = t0:Ts:tf;
N = numel(t);

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

P = zeros(4 , 4 , N);
P(:,:,N) = H ; 

%%% backward upgrade for P
tic
for k = N-1:-1:2
    P(:,:,k-1) = P(: , : , k) + Ts*(Q -P(: , : , k)*B*R^(-1)*B'*P(: , : , k)+P(: , : , k)*A+A'*P(: , : , k)) ;
end
disp(['Elapsed Time = ' num2str(toc)]);

%%% Plot Pi (P is symmetric ---> P12 = P21 , ...)

P11 = P(1 , 1 , :); P11 = P11(:);
P12 = P(1 , 2 , :); P12 = P12(:);
P13 = P(1 , 3 , :); P13 = P13(:);
P14 = P(1 , 4 , :); P14 = P14(:);

P22 = P(2 , 2 , :); P22 = P22(:);
P23 = P(2 , 3 , :); P23 = P23(:);
P24 = P(2 , 4 , :); P24 = P24(:);

P33 = P(3 , 3 , :); P33 = P33(:);
P34 = P(3 , 4 , :); P34 = P34(:);

P44 = P(4 , 4 , :); P44 = P44(:);

Fig1 = figure(1) ;
Fig1.Color = [0.9 0.9 0.9];
subplot(521)
plot(1:N ,P11,'b', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P11');

subplot(522)
plot(1:N ,P12,'b', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P12');

subplot(523)
plot(1:N ,P13,'b', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P13');

subplot(524)
plot(1:N ,P14,'b', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P14');

subplot(525)
plot(1:N ,P22,'r', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P22');

subplot(526)
plot(1:N ,P23,'r', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P23');

subplot(527)
plot(1:N ,P24,'r', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P24');

subplot(528)
plot(1:N ,P33,'m', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P33');

subplot(529)
plot(1:N ,P34,'m', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P34');

subplot(5,2,10)
plot(1:N ,P44,'k', 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
ylabel('P44');

disp('P Matrix obtained from "lqr" formula:')

% compute the steady state gains using lqr formula
[K_LQR,S,e] = lqr(A,B,Q,R,zeros(4,2));
disp(S)
disp('P Matrix obtained from the algorithm:')
disp(P(:,:,1))

%% extract optimal state feedback gain
K = zeros(2,4,N);

for o = 1:N-1
   
    K(:,:,o) = -inv(R) * B' * P (:,:,N-o);
    
end

K11 = K(1 ,1 ,:) ; K11 = K11(:) ;
K12 = K(1 ,2 ,:) ; K12 = K12(:) ;
K13 = K(1 ,3 ,:) ; K13 = K13(:) ;
K14 = K(1 ,4 ,:) ; K14 = K14(:) ;

K21 = K(2 ,1 ,:) ; K21 = K21(:) ;
K22 = K(2 ,2 ,:) ; K22 = K22(:) ;
K23 = K(2 ,3 ,:) ; K23 = K23(:) ;
K24 = K(2 ,4 ,:) ; K24 = K24(:) ;

Fig2 = figure(2);
Fig2.Color = [0.9 0.9 0.9];
subplot(2,1,1)
plot(1:o+1 ,K11 , 1:o+1 , K12 , 1:o+1 ,K13 , 1:o+1 ,K14 , 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
xlim([1 1499])
ylabel('K1');
legend('K11' , 'K12' , 'K13' , 'K14')

subplot(2,1,2)
plot(1:o+1 ,K21 , 1:o+1 , K22 , 1:o+1 ,K23 , 1:o+1 ,K24 , 'LineWidth' , 1.5);
grid on
xlabel('Iteration');
xlim([1 1499])
ylabel('K2');
legend('K21' , 'K22' , 'K23' , 'K24')


%% findinbg optimal control input
x = zeros(4 , N) ; 
x(: , 1) = [0;1;-1;5]; % Initial Condition
u = zeros(2 , 1 , N) ;

for i = 1:N-1
    u(:,:,i) = -R^(-1)*B'*P(:,:,i)*x(: ,i);
    x(: , i+1) = x(: , i) + Ts*(A*x(: , i) + B*u(:,:,i)); 
end

u1 = u(1,:,:);
u1 = u1(:);

u2 = u(2,:,:);
u2 = u2(:);


Fig3 = figure(3) ;
Fig3.Color = [0.9 0.9 0.9];
subplot(311);
plot(t ,x , 'LineWidth' , 1.5); hold on
grid on
xlabel('Time');
ylabel('X');
xlim([0 5])
ylim([-3 8])
legend('X1' , 'X2' , 'X3' , 'X4')


subplot(312);
plot(t ,u1 , 'b' ,'LineWidth' , 1.5); 
grid on
xlabel('Time');
ylabel('U');
legend('U1')
xlim([0 5])

subplot(313)
plot(t ,u2 ,'r', 'LineWidth' , 1.5); 
grid on
xlabel('Time');
ylabel('U');
legend('U2')
xlim([0 5])

%% total optimal cost

disp('The Steady State gains of LQR obtained from LQR formula:')
disp(-K_LQR)
disp('Gains obtained from HJB algorithm:')
disp(K(:,:,1500))
cost = 0.5 * x(:,1)' * S * x(:,1)