function signals_filtered = filter_freq_tanh(signals, Fs, f_up, df_up, f_down, df_down)
% make filtration in frequency domain by applying filtr with form of tanh 
% signals - siganal (it time domain from 0s till ...s) to be filtered 
% Fs - sampling frequency
% f_up, f_down - up and down limits of filtration
% df_up, df_douwn - width of filtration close to f_up, f_down

nt = size(signals,2) ;
nm = size(signals,1) ; 
            
f_array = (0:nt-1) * Fs/nt ;             
sp = ifft(signals, [], 2) ; 
           
mask_1 = 1/2 * ( tanh((f_array - f_down)/df_down) + 1)  +  1/2 * (1- tanh((f_array - (Fs-f_down))/df_down) ) -1 ; % 100 20           
mask_2 = 2-1/2 * ( tanh((f_array - f_up)/df_up) + 1)  -  1/2 * (1- tanh((f_array - (Fs-f_up))/df_up) )  ;           
frequency_mask = mask_1 .* mask_2 ; 
% figure;plot(f_array-Fs/2,fftshift(frequency_mask));title('frequency mask')

sp = sp .* (ones(nm,1) * frequency_mask) ; 
signals_filtered = real(fft(sp,[],2))  ;

            