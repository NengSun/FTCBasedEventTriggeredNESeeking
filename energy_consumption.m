clc
clear all
close all

%% setting of the game
N = 4; % number of players

% communication graph 
A = [0, 1, 0, 1;
    1, 0, 1, 0;
    0, 1, 0, 1;
    1, 0, 1, 0]; % adjacent matrix

B = A';
B = diag(B(:)); % B matrix

D = zeros(N); % degree matrix
for i = 1:N
    D(i,i) = sum(A(i,:));
end

L = D - A; % laplacian matrix

% gradient of the cost function 
H = [2.2, 0.1, 0.1, 0.1;
    0.1, 2.2, 0.1, 0.1;
    0.1, 0.1, 2.2, 0.1;
    0.1, 0.1, 0.1, 2.2];
h = -[20; 30; 40; 50]; 

%% design parameters in the proposed method
c1 = 10;
c2 = 11;
c3 = 10;
c4 = 2;
c5 = 0.6;
c6 = 0.1;
eta = 9;
q = 7/5;
theta = 2;
rho = 0.1;
dv = ones(N^2, 1) * 20; % dynamic variable
delta = 0.5;
beta = 10000000;

kappa = 0.99;
zeta = 0.1;

% condition check for Thm. 1
flag = false; 
flag1 = 2 * sqrt(c3 * c6 * (1 - rho)) - (1 - delta) / theta;
flag2 = c2 * (1 - rho) - eta;
if flag1 >= 0 && flag2 >0
    flag = true; % if the conditions are satisfied, flag = true
end

%% simulation startup
dt = 0.00001; % sampling time
T = [0:dt:2.3]; % simulation time sequence
t = 0; % current time
x = [22; 22; 22; 22]; % initial condition of action x
x_new = x; % temporal variable to store updated x
x_event = repmat(x,1,N^2); % value of x at the triggering instant
y = [-10; -10; -10; -10;
    -5; -5; -5; -5;
    0; 0; 0; 0;
    5; 5; 5; 5]; % initial condition of estimate y
y_new = y; % temporal variable to store updated y
y_event = repmat(y,1,N^2); % value of y at the triggering instant
% store x and y at each time step for plotting
X = [x]; 
Y = [y];

% store the triggering instants
t_event = cell(N^2,1);
for i = 1:N^2
    t_event{i} = [0];
end

% calculate gamma_ij at t^{ij}_k
gamma_event = zeros(N^2,1);
for i = 1:N
    for j = 1:N
        gamma_event((i-1)*N+j) = A(i,j) * (y_event((i-1)*N+j,(i-1)*N+j) - x_event(j,(i-1)*N+j));
        for m = 1:N
            gamma_event((i-1)*N+j) = gamma_event((i-1)*N+j) + A(i,m) * ...
                (y_event((i-1)*N+j,(i-1)*N+j) - y_event((m-1)*N+j,(i-1)*N+j));
        end
    end
end

% calculate ||yi-x||
est_err_i = zeros(N,1);
for i = 1:N
    est_err_i(i) = norm(y((i-1)*N+1:i*N) - x);
end
EST_Err_i = [est_err_i];

% error to Nash eq
x_eq = - inv(H) * h;
err = norm(x-x_eq);
Err = [err];
        

%% main body
for t_index = 1:length(T)-1
    
    %calculate gamma(t)
    gamma = (kron(L,eye(N)) + B)*(y-kron(ones(N,1),x));

    for i = 1:N
        for j = 1:N
            % measurement error E_ij(t)
            Eij = c1 * abs(gamma_event((i-1)*N+j))^q * sign(gamma_event((i-1)*N+j)) + ...
                c2 * tanh(beta * gamma_event((i-1)*N+j)) + c3 * gamma_event((i-1)*N+j) - ...
                c1 * abs(gamma((i-1)*N+j))^q * sign(gamma((i-1)*N+j)) - ...
                c2 * tanh(beta * (gamma((i-1)*N+j))) - c3 * gamma((i-1)*N+j);
            
            % triggering function
            gij = theta * (abs(Eij) - rho * c3 * abs(gamma((i-1)*N+j)) - rho * c2 - ...
                rho * c1 * abs(gamma((i-1)*N+j))^q);

            % if triggered
            if gij > dv((i-1)*N+j)
                y_event(:,(i-1)*N+j) = y; % update the value of x at the triggering instant
                x_event(:,(i-1)*N+j) = x; % update the value of y at the triggering instant
                t_event{(i-1)*N+j} = [t_event{(i-1)*N+j}, t]; % add the current time to 
                % the triggering instant sequence
                % calculate gamma_{ij} at the latest triggering instant
                gamma_event((i-1)*N+j) = A(i,j) * (y_event((i-1)*N+j,(i-1)*N+j) ...
                    - x_event(j,(i-1)*N+j)); 
                for m = 1:N
                    gamma_event((i-1)*N+j) = gamma_event((i-1)*N+j) + A(i,m) * ...
                        (y_event((i-1)*N+j,(i-1)*N+j) - y_event((m-1)*N+j,(i-1)*N+j));
                end
            end

            % update dynamic variable
            dv_dot = delta * abs(gamma((i-1)*N+j)) * (rho * c1 * abs(gamma((i-1)*N+j))^q + ...
                rho * c2 + rho * c3 * abs(gamma((i-1)*N+j)) - abs(Eij)) - ...
                c4 * dv((i-1)*N+j)^((q+1)/2) - c5 * dv((i-1)*N+j)^(1/2) - c6 * dv((i-1)*N+j)^2;
            dv((i-1)*N+j) = dv((i-1)*N+j) + dv_dot * dt;

        end
    end
    
    % update y
    y_dot = -c1 * abs(gamma_event).^q .* sign(gamma_event) - c2 * tanh(beta * gamma_event) ...
        - c3 * gamma_event;
    y_new = y + y_dot * dt;
    
    % update x
    grad = diag(H * reshape(y,N,N)) + h;
    x_dot = -eta * sign(grad);
    x_new = x + x_dot * dt;

    x = x_new;
    y = y_new;

    % calculate ||yi-x||
    est_err_i = zeros(N,1);
    for i = 1:N
        est_err_i(i) = norm(y((i-1)*N+1:i*N) - x);
    end
    % calculate ||x-x*||
    err = norm(x-x_eq);

    % append current state
    X = [X, x];
    Y = [Y, y];
    EST_Err_i = [EST_Err_i, est_err_i];
    Err = [Err, err];

    t = t + dt; % update current time



end



%% calculate the remaining error of x with respect to x*

alpha1 = min(c1 * (1 - rho) * N^(1-q) * (2 * min(eig(kron(L, eye(N)) + B)))^((q+1)/2), ...
    c4 * N^(1-q));
alpha2 = min((c2 * (1 - rho) - eta) * (2 * min(eig(kron(L, eye(N)) + B)))^0.5, c5);
tau = sqrt(2 / min(eig(kron(L, eye(N)) + B))) * min(c2 * N^2 * 0.2785 / ...
    (alpha2 * beta * (1-kappa)), (c2 * N^2 * 0.2785 / (2^((1-q)/2) * alpha1 * beta * ...
    (1-kappa)))^(1/(q+1)));

Hi = [];
for i = 1:N
    Hi = H(i,:);
    Hi = [Hi, norm(H(i,:))];
end
H_bar = max(Hi);

dev = 2*N*tau*H_bar*sqrt(max(eig(inv(H))))/(sqrt(min(eig(inv(H))))*min(eig(H))*(1-zeta));


%% plot

x_eq = repmat(x_eq, 1, length(T));

figure (1)
plot(T, X(1,:), 'r', T, X(2,:), 'b', T, X(3,:), 'g', T, X(4,:), 'c', 'LineWidth', 2)
hold on
plot(T, x_eq(1,:), 'r--', T, x_eq(2,:), 'b--', T, x_eq(3,:), 'g--', T, x_eq(4,:), 'c--', ...
    'LineWidth', 2)
legend('x_1', 'x_2', 'x_3', 'x_4', 'x^*_1', 'x^*_2', 'x^*_3', 'x^*_4', 'NumColumns', 2, ...
    'Location', 'SouthWest')
xlim([0,T(end)])
ylabel("The Players' Actions")
xlabel('Time')
set(gca,'XTick',[0:0.5:T(end),T(end)],'FontSize',15)
hold off

figure (2)
plot(T, Err, 'LineWidth', 1.5)
hold on
plot(T, ones(1, length(T)) * dev, 'k--', 'LineWidth', 1.5)
xlim([0,T(end)])
set(gca,'XTick',[0:0.5:T(end),T(end)],'FontSize',15)
ylabel({'$||x-x^*||$'},'Interpreter','latex')
xlabel('Time')
hold off

axes('Position',[0.48,0.55,0.38,0.3]); 
 plot(T, Err, 'LineWidth', 1.5)
 hold on
 plot(T, ones(1, length(T)) * dev, 'k--', 'LineWidth', 1.5)
 xlim([T(end)-0.55, T(end)])
 hold off

figure (3)
subplot(4,1,1)
plot(T, X(1,:), T, Y(1,:), T, Y(5,:), T, Y(9,:), T, Y(13,:),T, x_eq(1,:), 'LineWidth', 3, ...
    'LineStyle', ':')
xlim([0,T(end)])
set(gca,'XTick',[0:0.5:T(end),T(end)], 'FontSize',8.5)
legend('x_1', 'y_{11}', 'y_{21}', 'y_{31}', 'y_{41}', 'x^*_1', 'NumColumns', 2, ...
    'location', 'EastOutside')
ylabel('Estimate on x_1')

axes('Position',[0.18,0.8,0.18,0.048]); 
 plot(T, X(1,:), T, Y(1,:), T, Y(5,:), T, Y(9,:), T, Y(13,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([0.22, 0.27])

axes('Position',[0.43,0.8,0.18,0.048]); 
 plot(T, X(1,:), T, Y(1,:), T, Y(5,:), T, Y(9,:), T, Y(13,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([1.7, 2])

subplot(4,1,2)
plot(T, X(2,:), T, Y(2,:), T, Y(6,:), T, Y(10,:), T, Y(14,:), T, x_eq(2,:), 'LineWidth', 3, ...
    'LineStyle', ':')
xlim([0,T(end)])
set(gca,'XTick',[0:0.5:T(end),T(end)], 'FontSize',8.5)
legend('x_2', 'y_{12}', 'y_{22}', 'y_{32}', 'y_{42}', 'x^*_2', 'NumColumns', 2, ...
    'location', 'EastOutside')
ylabel('Estimate on x_2')

axes('Position',[0.18,0.59,0.18,0.048]); 
 plot(T, X(2,:), T, Y(2,:), T, Y(6,:), T, Y(10,:), T, Y(14,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([0.22, 0.27])

axes('Position',[0.43,0.59,0.18,0.048]); 
  plot(T, X(2,:), T, Y(2,:), T, Y(6,:), T, Y(10,:), T, Y(14,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([1.7, 2])

subplot(4,1,3)
plot(T, X(3,:), T, Y(3,:), T, Y(7,:), T, Y(11,:), T, Y(15,:), T, x_eq(3,:), 'LineWidth', 3, ...
    'LineStyle', ':')
xlim([0,T(end)])
set(gca,'XTick',[0:0.5:T(end),T(end)], 'FontSize',8.5)
legend('x_3', 'y_{13}', 'y_{23}', 'y_{33}', 'y_{43}', 'x^*_3', 'NumColumns', 2, ...
    'location', 'EastOutside')
ylabel('Estimate on x_3')

axes('Position',[0.18,0.37,0.18,0.048]); 
 plot(T, X(3,:), T, Y(3,:), T, Y(7,:), T, Y(11,:), T, Y(15,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([0.22, 0.27])

axes('Position',[0.43,0.37,0.18,0.048]); 
  plot(T, X(3,:), T, Y(3,:), T, Y(7,:), T, Y(11,:), T, Y(15,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([1.7, 2])

subplot(4,1,4)
plot(T, X(4,:), T, Y(4,:), T, Y(8,:), T, Y(12,:), T, Y(16,:), T, x_eq(4,:), 'LineWidth', 3, ...
    'LineStyle', ':')
xlim([0,T(end)])
set(gca,'XTick',[0:0.5:T(end),T(end)], 'FontSize',8.5)
legend('x_4', 'y_{14}', 'y_{24}', 'y_{34}', 'y_{44}', 'x^*_4', 'NumColumns', 2, ...
    'location', 'EastOutside')
xlabel('Time')
ylabel('Estimate on x_4')

axes('Position',[0.18,0.15,0.18,0.048]); 
 plot(T, X(4,:), T, Y(4,:), T, Y(8,:), T, Y(12,:), T, Y(16,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([0.22, 0.27])

axes('Position',[0.43,0.15,0.18,0.048]); 
  plot(T, X(4,:), T, Y(4,:), T, Y(8,:), T, Y(12,:), T, Y(16,:), 'LineWidth', 3, 'LineStyle', ':')
 xlim([1.7, 2])

figure (4)
% for player 1
subplot(4,1,1)
plot(t_event{1}, ones(1,length(t_event{1})), 'o', ...
    t_event{2}, ones(1,length(t_event{2}))*2, 'p', ...
    t_event{3}, ones(1,length(t_event{3}))*3, 'x', ...
    t_event{4}, ones(1,length(t_event{4}))*4, '*')
title('Triggering Instants of Player 1 for Each Estimate')
xlim([0,0.25])
set(gca, 'FontSize',8.5)

% for player 2
subplot(4,1,2)
plot(t_event{5}, ones(1,length(t_event{5})), 'o', ...
    t_event{6}, ones(1,length(t_event{6}))*2, 'p', ...
    t_event{7}, ones(1,length(t_event{7}))*3, 'x', ...
    t_event{8}, ones(1,length(t_event{8}))*4, '*')
title('Triggering Instants of Player 2 for Each Estimate')
xlim([0,0.25])
set(gca, 'FontSize',8.5)

% for player 3
subplot(4,1,3)
plot(t_event{9}, ones(1,length(t_event{9})), 'o', ...
    t_event{10}, ones(1,length(t_event{10}))*2, 'p', ...
    t_event{11}, ones(1,length(t_event{11}))*3, 'x', ...
    t_event{12}, ones(1,length(t_event{12}))*4, '*')
title('Triggering Instants of Player 3 for Each Estimate')
xlim([0,0.25])
set(gca, 'FontSize',8.5)

% for player 4
subplot(4,1,4)
plot(t_event{13}, ones(1,length(t_event{13})), 'o', ...
    t_event{14}, ones(1,length(t_event{14}))*2, 'p', ...
    t_event{15}, ones(1,length(t_event{15}))*3, 'x', ...
    t_event{16}, ones(1,length(t_event{16}))*4, '*')
title('Triggering Instants of Player 4 for Each Estimate')
xlim([0,0.25])
set(gca, 'FontSize',8.5)
xlabel('Time')
