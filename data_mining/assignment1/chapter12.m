%% clear environment variables
close all;
clear;
clc;
format compact;
%% extract data

% load test data wine, which including data matrix:classnumber = 3,wine:178*13,wine_labes:178*1
load chapter12_wine.mat;

% draw teting data's box videotex
figure;
boxplot(wine,'orientation','horizontal','labels',categories);
title('wine data_s box videotex','FontSize',12);
xlabel('attribute value','FontSize',12);
grid on;

% draw testing data's detached videotex
figure
subplot(3,5,1);
hold on
for run = 1:178
    plot(run,wine_labels(run),'*');
end
xlabel('sample','FontSize',10);
ylabel('classified label','FontSize',10);
title('class','FontSize',10);
for run = 2:14
    subplot(3,5,run);
    hold on;
    str = ['attrib ',num2str(run-1)];
    for i = 1:178
        plot(i,wine(i,run-1),'*');
    end
    xlabel('Sample','FontSize',10);
    ylabel('Attribute Value','FontSize',10);
    title(str,'FontSize',10);
end

% choose training set and testing set

% let first class 1-30,second class 60-95,third class 131-153 as training set
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% homologous training sets' labels also need to be seperated
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% Let first class 31-59,second class 96-130,third class 154-178 as testing set
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% homologous testing sets' labels also need to be seperated
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%% previous deal with data
% previous deal with data,let training set and testing set normalize to [0,1]

[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmax is a normalization function in matlab
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% SVM website training
model = svmtrain(train_wine_labels, train_wine, '-c 2 -g 1');
%% SVM网络预测
[predict_label, accuracy] = svmpredict(test_wine_labels, test_wine, model);

%% 结果分析

% 测试集的实际分类和预测分类图
% 通过图可以看出只有一个测试样本是被错分的
figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on;