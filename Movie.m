%movie
fig = figure('position',[100 100 1500 700]);

% calc movie:
for timestep=1:984
plot(TEMP_everytime_CAL((1:3963),timestep),'k');
hold on
set(gca,'ylim',[0 40])
title([num2str(timestep)]);
hold off
f(timestep) = getframe(fig);
end

% play movie:
close all
[h, w, p] = size(f(1).cdata);  % use 1st frame to get dimensions
hf = figure; 
% resize figure based on frame's w x h, and place at (150, 150)
set(hf, 'position', [100 100 w h]);
axis off
movie(h,hf,10,10,loc);