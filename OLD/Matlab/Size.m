%Load data
%X is the size of an eye
%Y is zero when on the picture, according to the label, the pupil is poorly
%detected and one, when a pupil is detected well
size = area2;

X = size(:,1);
y = size(:,2);

X_good = X(y==1);
average = mean(X_good)