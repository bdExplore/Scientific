% fid=fopen('2023.03.31-12.13.01_channel_0.dat','r'); % sin 62 dB - calibration
 %fid=fopen('2023.03.31-12.06.28_channel_0.dat','r'); % 70 dB big drone
%fid=fopen('2023.03.31-12.04.06_channel_0.dat','r'); % 60 dB big drone
 %fid=fopen('2023.03.31-12.02.41_channel_0.dat','r'); % 50 dB big drone
 fid=fopen('2023.03.31-12.01.01_channel_0.dat','r'); % 40 dB big drone
 %fid=fopen('2023.03.31-11.59.31_channel_0.dat','r'); % 0 dB big drone
% fid=fopen('2023.03.31-11.52.03_channel_0.dat','r'); % camera
%fid=fopen('2023.03.31-11.50.26_channel_0.dat','r'); % big drone 70 dB
% fid=fopen('2023.03.31-11.44.02_channel_0.dat','r'); % 0 small
% fid=fopen('2023.03.31-11.42.29_channel_0.dat','r'); % 70 small
%fid=fopen('2023.03.31-11.39.56_channel_0.dat','r'); % 60 small
%fid=fopen('2023.03.31-11.38.34_channel_0.dat','r'); % 50 small
%fid=fopen('2023.03.31-11.36.32_channel_0.dat','r'); % 40 small
data=fread(fid,'double');
fclose(fid);
if rem(length(data),2) == 0
    data = data(1 : end - 1);
end

data111 = transpose(data);
dataa = data111;