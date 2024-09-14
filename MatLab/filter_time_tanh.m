function [signals_filtered, signals_filtered_zeros] = filter_time_tanh(signals, Fs, t_up, dt_up, t_down, dt_down)
% masking in time domain by applying filtr with form of tanh 
% signals - signal (in time domain from 0s till ...s) to be filtered 
% Fs - sampling frequency
% t_up, t_down - up and down limits of masking
% dt_up, dt_douwn - width of masking close to t_up, t_down

nt = size(signals,2) ;
nm = size(signals,1) ; 
            
t_array = (0:nt-1) /Fs ;             
           
mask_1 = 1/2 * ( tanh((t_array - t_up)/dt_up) + 1);
mask_2 = 1-1/2 * ( tanh((t_array - t_down)/dt_down) + 1) ;     

time_mask = mask_1 .* mask_2 ; 
figure;plot(t_array,time_mask);title('time mask')

signals_filtered_zeros = signals .* (ones(nm,1) * time_mask) ; % masked siganls, zeros are outside the mask

signals_filtered = signals_filtered_zeros(:, time_mask > 1e-10); % only signals inside the mask

            