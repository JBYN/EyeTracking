%Load data
%LED is a list of the position of the LED
%CORNER is a list of the position of the corners
pos = ReferenceRD;
LED = pos(:,1:2);
CORNER = pos(:,3:4);
EYE = pos(:,5:6);

l = length(LED);
LED1 = zeros(l/2,2);
LED2 = zeros(l/2,2);
CORNER1 = zeros(l/2,2);
CORNER2 = zeros(l/2,2);
EYE1 = zeros(l/2,2); 
EYE2 = zeros(l/2,2);
j = 1;
for i = 1:2:(l-1)
    LED1(j,:) = LED(i,:);
    LED2(j,:) = LED(i+1,:);
    CORNER1(j,:) = CORNER(i,:);
    CORNER2(j,:) = CORNER(i+1,:);
    EYE1(j,:) = EYE(i,:);
    EYE2(j,:) = EYE(i+1,:);
    j = j+1;
end

%Determine the distance to the reference/led
CORNER_REF_1_X = abs(LED1(:,1)-CORNER1(:,1));
CORNER_REF_1_Y = abs(LED1(:,2)-CORNER1(:,2));
CORNER_REF_2_X = abs(LED2(:,1)-CORNER2(:,1));
CORNER_REF_2_Y = abs(LED2(:,2)-CORNER2(:,2));
CORNER_REF_1 = sqrt((LED1(:,1)-CORNER1(:,1)).^2 + (LED1(:,2)-CORNER1(:,2)).^2); 
CORNER_REF_2 = sqrt((LED2(:,1)-CORNER2(:,1)).^2 + (LED2(:,2)-CORNER2(:,2)).^2);

EYE_REF_1_X = abs(LED1(:,1)-EYE1(:,1));
EYE_REF_1_Y = abs(LED1(:,2)-EYE1(:,2));
EYE_REF_2_X = abs(LED2(:,1)-EYE2(:,1));
EYE_REF_2_Y = abs(LED2(:,2)-EYE2(:,2));
EYE_REF_1 = sqrt((LED1(:,1)-EYE1(:,1)).^2 + (LED1(:,2)-EYE1(:,2)).^2);
EYE_REF_2 = sqrt((LED2(:,1)-EYE2(:,1)).^2 + (LED2(:,2)-EYE2(:,2)).^2);

EYE_CORNER_1_X = abs(CORNER1(:,1)-EYE1(:,1));
EYE_CORNER_1_Y = abs(CORNER1(:,2)-EYE1(:,2));
EYE_CORNER_2_X = abs(CORNER2(:,1)-EYE2(:,1));
EYE_CORNER_2_Y = abs(CORNER2(:,2)-EYE2(:,2));
EYE_CORNER1 = sqrt((EYE1(:,1)-CORNER1(:,1)).^2 + (EYE1(:,2)-CORNER1(:,2)).^2);
EYE_CORNER2 = sqrt((EYE2(:,1)-CORNER2(:,1)).^2 + (EYE2(:,2)-CORNER2(:,2)).^2);

%remove outliers
[CORNER_REF_1_X,cr_1_x]= hampel(CORNER_REF_1_X,250,2);
[CORNER_REF_1_Y,cr_1_y]= hampel(CORNER_REF_1_Y,250,2);
[CORNER_REF_2_X,cr_2_x]= hampel(CORNER_REF_2_X,250,2);
[CORNER_REF_2_Y,cr_2_y]= hampel(CORNER_REF_2_Y,250,2);
[CORNER_REF_1,cr_1]= hampel(CORNER_REF_1,250,2);
[CORNER_REF_2,cr_2]= hampel(CORNER_REF_2,250,2);

[EYE_REF_1_X,er_1_x]= hampel(EYE_REF_1_X,250,2);
[EYE_REF_1_Y,er_1_y]= hampel(EYE_REF_1_Y,250,2);
[EYE_REF_2_X,er_2_x]= hampel(EYE_REF_2_X,250,2);
[EYE_REF_2_Y,er_2_y]= hampel(EYE_REF_2_Y,250,2);
[EYE_REF_1,er_1]= hampel(EYE_REF_1,250,2);
[EYE_REF_2,er_2]= hampel(EYE_REF_2,250,2);

[EYE_CORNER_1_X,ec_1_x]= hampel(EYE_CORNER_1_X,250,2);
[EYE_CORNER_1_Y,ec_1_y]= hampel(EYE_CORNER_1_Y,250,2);
[EYE_CORNER_2_X,ec_2_x]= hampel(EYE_CORNER_2_X,250,2);
[EYE_CORNER_2_Y,ec_2_y]= hampel(EYE_CORNER_2_Y,250,2);
[EYE_CORNER1,ec_1]= hampel(EYE_CORNER1,250,2);
[EYE_CORNER2,ec_2]= hampel(EYE_CORNER2,250,2);

%get number of outliers
cr_1_x = length(cr_1_x(cr_1_x==1));
cr_1_y = length(cr_1_y(cr_1_y==1));
cr_2_x = length(cr_2_x(cr_2_x==1));
cr_2_y = length(cr_2_y(cr_2_y==1));
cr_1 = length(cr_1(cr_1==1));
cr_2 = length(cr_2(cr_2==1));

er_1_x = length(er_1_x(er_1_x==1));
er_1_y = length(er_1_y(er_1_y==1));
er_2_x = length(er_2_x(er_2_x==1));
er_2_y = length(er_2_y(er_2_y==1));
er_1 = length(er_1(er_1==1));
er_2 = length(er_2(er_2==1));

ec_1_x = length(ec_1_x(ec_1_x==1));
ec_1_y = length(ec_1_y(ec_1_y==1));
ec_2_x = length(ec_2_x(ec_2_x==1));
ec_2_y = length(ec_2_y(ec_2_y==1));
ec_1 = length(ec_1(ec_1==1));
ec_2 = length(ec_2(ec_2==1));

%calculate the mean values
C_R_M_1_X = mean(CORNER_REF_1_X);
C_R_M_1_Y = mean(CORNER_REF_1_Y);
C_R_M_2_X = mean(CORNER_REF_2_X);
C_R_M_2_Y = mean(CORNER_REF_2_Y);
C_R_M_1 = mean(CORNER_REF_1);
C_R_M_2 = mean(CORNER_REF_2);

E_R_M_1_X = mean(EYE_REF_1_X);
E_R_M_1_Y = mean(EYE_REF_1_Y);
E_R_M_2_X = mean(EYE_REF_2_X);
E_R_M_2_Y = mean(EYE_REF_2_Y);
E_R_M_1 = mean(EYE_REF_1);
E_R_M_2 = mean(EYE_REF_2);

E_C_M_1_X = mean(EYE_CORNER_1_X);
E_C_M_1_Y = mean(EYE_CORNER_1_Y);
E_C_M_2_X = mean(EYE_CORNER_2_X);
E_C_M_2_Y = mean(EYE_CORNER_2_Y);
E_C_M_1 = mean(EYE_CORNER1);
E_C_M_2 = mean(EYE_CORNER2);

%calculate the standard deviations
C_R_S_1_X = std(CORNER_REF_1_X);
C_R_S_1_Y = std(CORNER_REF_1_Y);
C_R_S_2_X = std(CORNER_REF_2_X);
C_R_S_2_Y = std(CORNER_REF_2_Y);
C_R_S_1 = std(CORNER_REF_1);
C_R_S_2 = std(CORNER_REF_2);

E_R_S_1_X = std(EYE_REF_1_X);
E_R_S_1_Y = std(EYE_REF_1_Y);
E_R_S_2_X = std(EYE_REF_2_X);
E_R_S_2_Y = std(EYE_REF_2_Y);
E_R_S_1 = std(EYE_REF_1);
E_R_S_2 = std(EYE_REF_2);

E_C_S_1_X = std(EYE_CORNER_1_X);
E_C_S_1_Y = std(EYE_CORNER_1_Y);
E_C_S_2_X = std(EYE_CORNER_2_X);
E_C_S_2_Y = std(EYE_CORNER_2_Y);
E_C_S_1 = std(EYE_CORNER1);
E_C_S_2 = std(EYE_CORNER2);
