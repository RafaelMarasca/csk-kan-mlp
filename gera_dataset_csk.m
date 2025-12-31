clear all; close all; clc

%% Configuration
N_train = 11264; % Number of training symbols

% --- Part 1: Define Specific Target Runs ---
% Format: [M, Var_dB]
specific_runs = [
    4, -40;
    4, -1000;
    8, -48;
    8, -1000
];

% --- Part 2: Define Sweep Runs ---
% Sweep from -20dB to -60dB with a step of 2dB
sweep_dB = -20:-2:-60; 
sweep_M  = [4, 8]; % Perform sweep for both 4-CSK and 8-CSK

%% Execution Loop
% We combine specific runs and sweep runs to process them efficiently.
% (You can comment out the sweep loop if you only want the specific files)

% 1. Process Specific Targets first
fprintf('--- Generating Specific Targets ---\n');
for i = 1:size(specific_runs, 1)
    run_generation(specific_runs(i, 1), N_train, specific_runs(i, 2), 'specific');
end

% 2. Process Sweep
fprintf('\n--- Generating Sweep Datasets ---\n');
for m = sweep_M
    for db = sweep_dB
        run_generation(m, N_train, db, 'sweep');
    end
end

disp('All datasets generated successfully.');


%% Helper Function to Generate and Save Data
function run_generation(M, N_train, Var_dB, type)
    
    % Generate the filename based on parameters
    % Example: data_4csk_m40dB.csv (using 'm' to represent minus)
    if Var_dB < 0
        sign_str = 'm';
    else
        sign_str = 'p';
    end
    
    % Construct filename: e.g., "data_4csk_m40dB.csv"
    filename = sprintf('data_%dcsk_%s%ddB.csv', M, sign_str, abs(Var_dB));
    
    fprintf('Generating: M=%d, Var_dB=%d ... ', M, Var_dB);

    % Call your custom function
    % Note: Ensuring the function call matches your original syntax
    [tx_simb, Vrgb, RGB, XYZ, xy, XYZne, xyne] = tx_rx_csk(M, N_train, Var_dB, 1);

    % Create Table
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

    % Write to CSV
    writetable(T_all, filename);
    
    fprintf('Saved to %s\n', filename);
end