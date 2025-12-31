% MATLAB Noise Analysis Experiment with fitcnet
clear; clc;

% Configuration
n_repeats = 10;
base_seed = 42;
train_size = 64;
test_size = 64;
output_file = 'matlab_noise_results.csv';

% Initialize results storage
% Columns: Dataset, Noise_Level_dB, Model, Mean_SER, Std_SER
results = {};
row_idx = 1;

fprintf('Starting MATLAB Noise Analysis Experiment...\n');
fprintf('==============================================\n');

% Define datasets and configurations
datasets = struct('name', {}, 'pattern', {}, 'features', {}, 'hidden_layers', {}, 'model_name', {});

% Configure FNN-1 (4CSK)
datasets(1).name = '4CSK';
datasets(1).pattern = 'data_4csk_m*dB.csv';
datasets(1).features = {'Vr', 'Vg', 'Vb'};
datasets(1).hidden_layers = 10;
datasets(1).model_name = 'FNN-1';

% Configure FNN-3 (8CSK)
datasets(2).name = '8CSK';
datasets(2).pattern = 'data_8csk_m*dB.csv';
datasets(2).features = {'X_ne', 'Y_ne', 'Z_ne'};
datasets(2).hidden_layers = 100;
datasets(2).model_name = 'FNN-3';

% Iterate over datasets
for d = 1:length(datasets)
    ds = datasets(d);
    fprintf('\nDataset: %s (Model: %s)\n', ds.name, ds.model_name);
    
    % Find files
    files = dir(ds.pattern);
    
    for f = 1:length(files)
        filename = files(f).name;
        
        % Skip the reference noiseless file (m1000dB)
        if contains(filename, 'm1000dB')
            continue;
        end
        
        % Extract noise level from filename (e.g., 'data_4csk_m20dB.csv' -> 20)
        tokens = regexp(filename, '_m(\d+)dB', 'tokens');
        if isempty(tokens)
            continue;
        end
        noise_level = -str2double(tokens{1}{1});
        
        fprintf('  Processing Noise Level: %ddB... ', noise_level);
        
        % Load full dataset
        full_table = readtable(filename);
        num_samples = height(full_table);
        
        if num_samples < (train_size + test_size)
            warning('Not enough samples in %s. Skipping.', filename);
            continue;
        end
        
        sers = zeros(n_repeats, 1);
        
        % Run independent repeats
        for run_idx = 1:n_repeats
            % 1. Set seed for this run (matches Python logic: base + run_idx)
            % Python range is 0-9, MATLAB uses 1-10 here, so we adjust to match seeds if strictly necessary.
            % To match Python's (base_seed + 0) to (base_seed + 9):
            current_seed = base_seed + (run_idx - 1);
            rng(current_seed);
            
            % 2. Split Data (Random Permutation)
            perm = randperm(num_samples);
            train_idx = perm(1:train_size);
            test_idx = perm(train_size+1 : train_size+test_size);
            
            train_data = full_table(train_idx, :);
            test_data = full_table(test_idx, :);
            
            % Extract Features and Targets
            X_train = train_data{:, ds.features};
            y_train = train_data.Symbol;
            X_test = test_data{:, ds.features};
            y_test = test_data.Symbol;
            
            % 3. Train Model (fitcnet)
            % Uses default parameters except for Activation and Layers
            Mdl = fitcnet(X_train, y_train, ...
                'LayerSizes', ds.hidden_layers, ...
                'Activations', 'relu');
            
            % 4. Predict and Calculate SER
            preds = predict(Mdl, X_test);
            ser = mean(preds ~= y_test);
            sers(run_idx) = ser;
        end
        
        % Aggregate results
        mean_ser = mean(sers);
        std_ser = std(sers);
        
        fprintf('Mean SER: %.4f\n', mean_ser);
        
        % Store in results cell array
        results(row_idx, :) = {ds.name, noise_level, ds.model_name, mean_ser, std_ser};
        row_idx = row_idx + 1;
    end
end

% Convert to Table and Save
T = cell2table(results, 'VariableNames', {'Dataset', 'Noise_Level_dB', 'Model', 'Mean_SER', 'Std_SER'});
writetable(T, output_file);

fprintf('\nSuccess! Results saved to %s\n', output_file);