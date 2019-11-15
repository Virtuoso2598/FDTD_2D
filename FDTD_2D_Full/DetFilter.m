det = load ('detect.txt');

% d = designfilt('lowpassiir','FilterOrder',500, ...
%     'HalfPowerFrequency',0.5,'DesignMethod','butter');
% filtred = filtfilt(d,abs(det(:,2)));
% plot(filtred)

tstrt=14000;

windowSize = 2048;
b = (1/windowSize)*ones(1,windowSize);
a=1;
x=abs(det(:,2));
filtred = filter(b,a,x);

subplot(1,2,1)
plot(det(1:tstrt,1),filtred(1:tstrt)); grid on

subplot(1,2,2)
plot(det(tstrt:end,1),filtred(tstrt:end)); grid on

%% ---------------------------------------------------
figure();
load detector_filtered_field.txt
contourf(detector_filtered_field)
colorbar
%% --------------------------------------------------
det_ARC = load('det_ARC.txt');
det_CF = load('det_CF.txt');
det_SD = load('det_SD.txt');

figure();
title('ARC');

subplot(1,3,1)
% title('ARC');
plot(det_ARC(:,1),det_ARC(:,3)); grid on;

subplot(1,3,2)
% title('CF');
plot(det_CF(:,1),det_CF(:,3)); grid on

subplot(1,3,3)
% title('SD');
plot(det_SD(:,1),det_SD(:,3)); grid on

%% ---------------------------------------------------------
d_ARC = load('ARC.txt');
d_CF = load('CF.txt');
d_SD = load('SD.txt');

figure();
title('ARC');

subplot(1,3,1)
% title('ARC');
plot(d_ARC(:,1),d_ARC(:,3)); grid on;

subplot(1,3,2)
% title('CF');
plot(d_CF(:,1),d_CF(:,3)); grid on

subplot(1,3,3)
% title('SD');
plot(d_SD(:,1),d_SD(:,3)); grid on

%% ----------------------------------------------------------------
RC_line = load('RC_line.txt');
figure();
plot(RC_line);
