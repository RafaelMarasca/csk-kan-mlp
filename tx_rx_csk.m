function [tx_simb Vrgb RGBl XYZl xyl XYZne xyne] = tx_rx_csk(M, Nsimb, Var_dB, aprende)
% 21/05/2024
% M         -> ordem do M-CSK
% Nsimb     -> nº de símbolos transmitidos
% Var_dB    -> variância do ruído em dB

% clear all; close all; clc;
% %tic
% M = 4;
% Nsimb = 10;
% Var_dB = -100;
% aprende = 0;

Pmax = 0.7;     % utilizaremos 70% da potência máxima de 1W

%% Constantes
H = [  143.38   4.8802  0.7780;     % matriz CSI de mathias_2021 
       2.8135   153.53  53.289;     % (foi modificada ordem dos elementos de BGR para RGB)
       1.7792   18.696  133.19];    % [mV/A]

% Matrizes de conversão
% https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
% RGB2XYZ = [ .4124  .3576  .1805;
%                 .2126  .7152  .0722;
%                 .0193  .1192  .9505]; % não ficou bom
% https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_RGB_color_space
RGB2XYZ = [ .49000  .31000  .20000;
            .17697  .81240  .01063;
            .00000  .01000  .99000]; % esta é da CIE-RGB

%Parâmetros para 4-CSK
P4csk = [ 0     1       0;      % S1 [0 0] % Matriz de potências
          0.358 0.410   0.232;  % S2 [0 1] 
          0     0       1;      % S3 [1 0]
          1     0       0]';    % S4 [1 1]
%Ptotal = sum(P4csk)
I4csk = [   -6.174e-2*P4csk(1,:).^2+4.855e-1*P4csk(1,:)-4.510e-3;
            -3.471e-2*P4csk(2,:).^2+3.586e-1*P4csk(2,:)-3.476e-3;
            -3.271e-2*P4csk(3,:).^2+3.471e-1*P4csk(3,:)-3.896e-3];
I4csk(I4csk<0)=0;   % (exclui as correntes negativas)
I4csk = I4csk*Pmax;  

% Parâmetros para 8-CSK
P8csk = [   0     1       0;        % Simbolo 1 (S1)
            0     0.800   0.200;    % S2
            0.317 0.683   0;        % S3
            0.643 0.295   0.062;    % S4
            0     0       1;        % S5
            0.086 0.417   0.497;    % S6
            0.572 0.034   0.394;    % S7
            1     0       0]';      % S8
%Ptotal = sum(P8csk)
I8csk = [   -6.174e-2*P8csk(1,:).^2+4.855e-1*P8csk(1,:)-4.510e-3;
            -3.471e-2*P8csk(2,:).^2+3.586e-1*P8csk(2,:)-3.476e-3;
            -3.271e-2*P8csk(3,:).^2+3.471e-1*P8csk(3,:)-3.896e-3];
I8csk(I8csk<0)=0;   % (exclui as correntes negativas)
I8csk = I8csk*Pmax;

% Ponto padrão de branco luz do dia
%wp = whitepoint('D65');wpMag = sum(wp,2); 
%x_wp = wp(:,1)./wpMag; y_wp = wp(:,2)./wpMag;

% Pontos da modulação CSK da norma IEEE 802.15.7-2011 
% Combinação de códigos de cores nº11 [100-010-001] (este é o que mais bateu com os nossos LEDs)
% Codigos binários: [0 0], [0 1], [1 0], [1 1]
cod4csk = [0 0; 0 1; 1 0; 1 1];
x_4csk = [.190 .327 .090 .700];
y_4csk = [.780 .403 .130 .300];
cod8csk = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
x_8csk = [.190 .157 .360 .491 .090 .186 .395 .700];
y_8csk = [.780 .563 .620 .414 .130 .329 .215 .300];

%% Gera os dados a serem transmitidos
if aprende
    simb = 0;
    for ns = 1:Nsimb
        tx_simb(ns) = simb;
        if simb == (M-1)
            simb = 0;
        else
            simb = simb+1;
        end
    end
else
    tx_simb = uint16(randi([0 M-1],1,Nsimb));
end

% Sequência dos sinais no tempo
if M == 4 
    for n=1:Nsimb
        switch tx_simb(n)
          case 0
            Irgb(:,n) = I4csk(:,1);
          case 1
            Irgb(:,n) = I4csk(:,2);
          case 2
            Irgb(:,n) = I4csk(:,3);
          case 3
            Irgb(:,n) = I4csk(:,4);
        end
    end
else
    for n=1:Nsimb
        switch tx_simb(n)
          case 0
            Irgb(:,n) = I8csk(:,1);
          case 1
            Irgb(:,n) = I8csk(:,2);
          case 2
            Irgb(:,n) = I8csk(:,3);
          case 3
            Irgb(:,n) = I8csk(:,4);
          case 4
            Irgb(:,n) = I8csk(:,5);
          case 5
            Irgb(:,n) = I8csk(:,6);
          case 6
            Irgb(:,n) = I8csk(:,7);
          case 7
            Irgb(:,n) = I8csk(:,8);
        end
    end
end

%% Canal
V = H*Irgb*1e-3; % [V]

% V - tensão na saída o TIA, Irgb - correntes aplicada aos LEDs
% o efeito da não linearidade teria que ser adicionada aqui (mas não exploraremos neste trabalho)
% adiciona ruído no sinal após o TIA para os triestímulos
Var = 10.^(Var_dB/10);
Noise = sqrt(Var)*randn(3,size(V,2));
Vrgb = V + Noise;
% var(Noise')

%% Com equalização
% Triestímulos RGB
RGBl = H^-1*Vrgb;       % (6)
% Triestímulos XYZ
XYZl = RGB2XYZ*RGBl;    % (7)
% Converte para o espaço de cor CIE x-y (1931)
soma = sum(XYZl);        
xyl = [ XYZl(1,:)./soma;  XYZl(2,:)./soma]; % (2)

%% Sem equalização
% Triestímulos XYZ
XYZne = RGB2XYZ*Vrgb;   % (7)
% Converte para o espaço de cor CIE x-y (1931)
soma = sum(XYZne);
xyne = [ XYZne(1,:)./soma;  XYZne(2,:)./soma];  % (2)













