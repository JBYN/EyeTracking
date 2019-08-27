%Load data
%X is the area of the white surface
%Y is zero when on the picture, according to the label, the pupil is poorly
%detected and one, when a pupil is detected well
area = area1;

Z = area(:,1)./area(:,2);
X = area(:,3);
y = area(:,4);

X_good = X(y==1);
Z1 = Z(y==1);
X_bad = X(y==0);
Z2 = Z(y==0);

%X_bad = [1 2 3 4 4 2 12 9 16 8];
%X_good = [25 28 30 17 28 26 20 24];
plot(X_good,Z1,'og');
hold on
plot(X_bad,Z2,'xr');
