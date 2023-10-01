% acronym definitions
clear;

acronyms = {};
for sample = 1:5
    range = {'96-00','01-05','06-10','11-15','16-20'};
    Data1 = readtable(strcat('CZ',range{sample},'.csv'),'ReadVariableNames',true);
    Date1 = unique(table2array(Data1(:,3)));
    N = table2array(Data1(1,12));
    Num(sample) = N;
    J = 10;
    T = 60;
    for n=1:N
        acronyms = [acronyms;char(Data1{(n-1)*600+1,5})];
    end
end
acronyms = unique(acronyms);
N = size(acronyms,1);
Data2 = readtable('SignalDoc.csv','ReadVariableNames',true);
tab = {};
for n=1:N
    ind = find(strcmp(Data2{:,"Acronym"},acronyms{n}));
    tab = [tab;Data2{ind,["LongDescription","Authors"]}];
end
tab = [acronyms,tab];
tab = cell2table(tab,"VariableNames",{'Acronym','Description','Authors'});
table2latex(tab(41:80,:))
