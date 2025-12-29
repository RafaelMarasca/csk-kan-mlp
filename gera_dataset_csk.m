clear all; close all; clc

Var_dB = -50;
%Var_dB = -47;
%Var_dB = -100;
%Var = 10.^(Var_dB/10);

M = 8;
N_train = 1024 + 64;  % nº de símbolos de aprendizagem

%% treino    
% Gera o dataset
%[tx_simba, Vrgba, RGBa, RGBeqa, xya, xyeqa] = tx_rx_csk_aprendizado(M, Nt, Var);
[tx_simb, Vrgb, RGB, XYZ, xy, XYZne, xyne] = tx_rx_csk(M, N_train, Var_dB,1);



T_all = table;
T_all.Vr   = Vrgb(1,:)';
T_all.Vg   = Vrgb(2,:)';
T_all.Vb   = Vrgb(3,:)';
T_all.R    = RGB(1,:)';
T_all.G    = RGB(2,:)';
T_all.B    = RGB(3,:)';
T_all.X    = XYZ(1,:)';
T_all.Y    = XYZ(2,:)';
T_all.Z    = XYZ(3,:)';
T_all.x    = xy(1,:)';
T_all.y    = xy(2,:)';
T_all.X_ne = XYZne(1,:)';
T_all.Y_ne = XYZne(2,:)';
T_all.Z_ne = XYZne(3,:)';
T_all.x_ne = xyne(1,:)';
T_all.y_ne = xyne(2,:)';
T_all.Symbol = categorical(tx_simb');

writetable(T_all,'all_data_8csk.csv');   % one CSV with all variables