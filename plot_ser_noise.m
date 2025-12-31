% Plot Consolidated Results (Python KAN/MLE + MATLAB FNN)
clear; clc; close all;

% --- Configuration ---
python_file = './results_refactored/noise_analysis_results.csv';
matlab_file = 'matlab_noise_results.csv';
output_dir = 'plots';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% --- 1. Load Data ---
fprintf('Loading data...\n');

% Load Python results (KAN, MLE)
T_py = readtable(python_file);
% Filter: Keep only KAN and MLE from Python
mask_py = ismember(T_py.Model, {'KAN', 'MLE'});
T_py = T_py(mask_py, :);

% Load MATLAB results (FNN)
T_mat = readtable(matlab_file);

% Combine tables
T_all = [T_py; T_mat];

% --- 2. Plotting ---
datasets = {'4CSK', '8CSK'};

for d = 1:length(datasets)
    ds_name = datasets{d};
    fprintf('Plotting %s...\n', ds_name);
    
    % Filter data for this dataset
    if iscell(T_all.Dataset)
        ds_mask = strcmp(T_all.Dataset, ds_name);
    else
        ds_mask = T_all.Dataset == string(ds_name);
    end
    data = T_all(ds_mask, :);
    
    % Prepare Figure
    f = figure('Name', ds_name, 'Color', 'w');
    set(f, 'Units', 'pixels', 'Position', [100 100 800 600]); % Slightly taller for clearer fonts
    hold on;
    
    % Define Model Specifics
    if strcmp(ds_name, '4CSK')
        models_to_plot = {'MLE', 'KAN', 'FNN-1'};
        legend_labels = containers.Map({'MLE', 'KAN', 'FNN-1'}, ...
                                       {'MLE', 'KAN-3', 'FNN-1'});
        markers = containers.Map({'MLE', 'KAN', 'FNN-1'}, {'o', 's', '^'});
        colors = containers.Map({'MLE', 'KAN', 'FNN-1'}, ...
                                {[0 0.6 0], [0 0.447 0.741], [0.85 0.325 0.098]});
    else
        models_to_plot = {'MLE', 'KAN', 'FNN-3'};
        legend_labels = containers.Map({'MLE', 'KAN', 'FNN-3'}, ...
                                       {'MLE', 'KAN-2', 'FNN-3'});
        markers = containers.Map({'MLE', 'KAN', 'FNN-3'}, {'o', 's', '^'});
        colors = containers.Map({'MLE', 'KAN', 'FNN-3'}, ...
                                {[0 0.6 0], [0 0.447 0.741], [0.85 0.325 0.098]});
    end
    
    % Loop through models
    for m = 1:length(models_to_plot)
        model_key = models_to_plot{m};
        
        % Filter model
        if iscell(data.Model)
            model_mask = strcmp(data.Model, model_key);
        else
            model_mask = data.Model == string(model_key);
        end
        
        sub_data = data(model_mask, :);
        
        if isempty(sub_data)
            warning('No data found for %s in %s', model_key, ds_name);
            continue;
        end
        
        % Sort by Noise Level
        sub_data = sortrows(sub_data, 'Noise_Level_dB');
        
        % Data
        x = sub_data.Noise_Level_dB;
        y = sub_data.Mean_SER * 100;
        
        % Plot
        plot(x, y, ...
            'DisplayName', legend_labels(model_key), ...
            'Marker', markers(model_key), ...
            'Color', colors(model_key), ...
            'LineWidth', 2, ... % Increased LineWidth for visibility
            'MarkerSize', 8, ... % Increased MarkerSize
            'MarkerFaceColor', colors(model_key));
    end
    
    % Formatting with Increased Font Sizes
    xlabel('Noise Variance [dBV^2]', 'FontSize', 14, 'Interpreter', 'tex', 'FontWeight', 'bold');
    ylabel('SER [%]', 'FontSize', 14, 'FontWeight', 'bold');
    
    grid on;
    ax = gca;
    ax.GridAlpha = 0.3;
    ax.FontSize = 12; % Tick Label Size
    ax.LineWidth = 1.2;
    
    % Legend: Fixed to Top-Left (Northwest) to stay on one side
    legend('show', 'Location', 'northwest', 'FontSize', 12);
    
    title('');
    hold off;
    
    % Save
    filename_png = fullfile(output_dir, sprintf('combined_noise_%s.png', ds_name));
    filename_pdf = fullfile(output_dir, sprintf('combined_noise_%s.pdf', ds_name));
    
    saveas(f, filename_png);
    exportgraphics(f, filename_pdf, 'ContentType', 'vector');
    
    fprintf('Saved plots to %s\n', filename_png);
end

fprintf('Done.\n');