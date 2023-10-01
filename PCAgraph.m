% PCA graph illustration
clear;clc;
rng('default')
mu = [0 0];
Sigma = [1 1.5; 1.5 3];
R = mvnrnd(mu, Sigma, 500);

[vec,val]=eig(R'*R/500);

val=diag(val);
val=val;

s= scatter(R(:,1),R(:,2),"filled");
alpha(s,0.5)

hold on

quiver(mu(1),mu(2),vec(1,2)*val(2),vec(2,2)*val(2),1.5,'red','LineWidth',2,'MaxHeadSize',0.5);
quiver(mu(1),mu(2),vec(1,1)*val(1),vec(2,1)*val(1),8,'red','LineWidth',2,'MaxHeadSize',2);

xlim([-5 5])
ylim([-5 5])

axis equal
grid on
box on
fontsize(gcf,14,'points')
% exportgraphics(gcf,'PCAgraph.pdf','BackgroundColor','none','ContentType','vector')