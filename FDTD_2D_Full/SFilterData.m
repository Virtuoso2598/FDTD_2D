x=detAll(:,4);

windowSize = 2048;
b = (1/windowSize)*ones(1,windowSize);

a = 1;

y = filter(b,a,x);

plot(x)
hold on
plot(y)

grid on
legend('Input Data','Filtered Data','Location','NorthWest')
title('Plot of Input and Filtered Data')
